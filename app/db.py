"""Supabase access. Uploads are versioned so a failed re-upload never loses data.

Two kinds of documents live side by side (documents.scope):
  core - the curated nutrition knowledge base, managed by the admin CLI, visible to everyone;
  user - a private upload, visible only to the owner_id that created it.
Isolation is enforced in the SQL function match_documents, not in the UI.

With VECTOR_STORE=local every function here is served by app/local_store.py instead (a file on
disk with the same semantics), so the app runs with no database at all.
"""
import logging
import uuid
from datetime import UTC, datetime

from supabase import Client, create_client

from app import local_store
from app.config import get_settings

log = logging.getLogger(__name__)

_sb: Client | None = None

DOC_COLUMNS = "doc_id,filename,num_chunks,scope,owner_id,updated_at"


def _local() -> bool:
    return get_settings().vector_store == "local"


def get_client() -> Client:
    global _sb
    if _sb is None:
        s = get_settings()
        _sb = create_client(s.supabase_url, s.supabase_service_role_key)
    return _sb


def replace_document(
    doc_id: str,
    filename: str,
    rows: list[dict],
    *,
    scope: str = "core",
    owner_id: str | None = None,
    file_hash: str | None = None,
) -> int:
    """Swap in a new version of a document without ever leaving it half-written.

    1. insert all new chunks under a fresh batch_id (invisible to search);
    2. on any failure, delete that partial batch and re-raise - the old version stays live;
    3. flip documents.active_batch to the new batch (single-row upsert = atomic);
    4. delete the previous batches (cleanup only; if this fails the data is merely stale).
    """
    if _local():
        return local_store.replace_document(doc_id, filename, rows, scope=scope, owner_id=owner_id, file_hash=file_hash)
    s = get_settings()
    sb = get_client()
    batch_id = str(uuid.uuid4())

    try:
        for i in range(0, len(rows), s.insert_batch_size):
            batch = [{**r, "doc_id": doc_id, "batch_id": batch_id} for r in rows[i : i + s.insert_batch_size]]
            sb.table("chunks").insert(batch).execute()
    except Exception:
        log.exception("insert failed for doc_id=%s, rolling back batch %s", doc_id, batch_id)
        sb.table("chunks").delete().eq("batch_id", batch_id).execute()
        raise

    sb.table("documents").upsert(
        {
            "doc_id": doc_id,
            "active_batch": batch_id,
            "filename": filename,
            "num_chunks": len(rows),
            "scope": scope,
            "owner_id": owner_id,
            "file_hash": file_hash,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    ).execute()

    try:
        sb.table("chunks").delete().eq("doc_id", doc_id).neq("batch_id", batch_id).execute()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        log.warning("old-batch cleanup failed for doc_id=%s (harmless, will be retried next upload)", doc_id)

    return len(rows)


def get_document(doc_id: str) -> dict | None:
    if _local():
        return local_store.get_document(doc_id)
    rows = get_client().table("documents").select(f"{DOC_COLUMNS},file_hash").eq("doc_id", doc_id).limit(1).execute().data
    return rows[0] if rows else None


def find_document_by_hash(file_hash: str, scope: str, owner_id: str | None) -> dict | None:
    """Same bytes already stored in the same space (core, or this owner's)? Then skip re-embedding."""
    if _local():
        return local_store.find_document_by_hash(file_hash, scope, owner_id)
    q = get_client().table("documents").select(DOC_COLUMNS).eq("file_hash", file_hash).eq("scope", scope)
    q = q.eq("owner_id", owner_id) if owner_id else q.is_("owner_id", "null")
    rows = q.limit(1).execute().data
    return rows[0] if rows else None


def list_documents(owner_id: str | None) -> dict[str, list[dict]]:
    """Core documents plus (if an owner is given) that owner's private ones."""
    if _local():
        return local_store.list_documents(owner_id)
    sb = get_client()
    cols = "doc_id,filename,num_chunks,updated_at"
    core = sb.table("documents").select(cols).eq("scope", "core").order("doc_id").execute().data
    mine: list[dict] = []
    if owner_id:
        mine = sb.table("documents").select(cols).eq("scope", "user").eq("owner_id", owner_id).order("updated_at", desc=True).execute().data
    return {"core": core, "mine": mine}


def count_user_documents(owner_id: str, excluding_doc_id: str | None = None) -> int:
    if _local():
        return local_store.count_user_documents(owner_id, excluding_doc_id)
    q = get_client().table("documents").select("doc_id", count="exact").eq("scope", "user").eq("owner_id", owner_id)
    if excluding_doc_id:
        q = q.neq("doc_id", excluding_doc_id)
    return q.limit(1).execute().count or 0


def delete_document(doc_id: str) -> None:
    """Remove the registry row first (instantly invisible to search), then the chunks."""
    if _local():
        return local_store.delete_document(doc_id)
    sb = get_client()
    sb.table("documents").delete().eq("doc_id", doc_id).execute()
    sb.table("chunks").delete().eq("doc_id", doc_id).execute()


def search_chunks(
    query_embedding: list[float],
    top_k: int,
    doc_id: str | None = None,
    min_similarity: float = 0.0,
    owner_id: str | None = None,
    scope: str = "all",
) -> list[dict]:
    if _local():
        return local_store.search_chunks(query_embedding, top_k, doc_id, min_similarity, owner_id, scope)
    payload = {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "filter_doc_id": doc_id,
        "min_similarity": min_similarity,
        "filter_owner": owner_id,
        "filter_scope": scope,
    }
    return get_client().rpc("match_documents", payload).execute().data or []


# --- interactions: answered questions + thumbs up/down (db/migrations/003_interactions.sql) ------

def log_interaction(row: dict) -> None:
    if _local():
        return local_store.log_interaction(row)
    get_client().table("interactions").upsert(row).execute()


def rate_interaction(request_id: str, owner_id: str | None, rating: int, comment: str | None) -> bool:
    """Only the owner who asked (or, for anonymous chats, anyone holding the unguessable id) may rate."""
    if _local():
        return local_store.rate_interaction(request_id, owner_id, rating, comment)
    q = get_client().table("interactions").update(
        {"rating": rating, "comment": comment, "rated_at": datetime.now(UTC).isoformat()}
    ).eq("request_id", request_id)
    q = q.eq("owner_id", owner_id) if owner_id else q.is_("owner_id", "null")
    return bool(q.execute().data)


def list_interactions(rated_only: bool = True, limit: int = 500) -> list[dict]:
    if _local():
        return local_store.list_interactions(rated_only, limit)
    q = get_client().table("interactions").select("*").order("created_at", desc=True).limit(limit)
    if rated_only:
        q = q.not_.is_("rating", "null")
    return q.execute().data or []
