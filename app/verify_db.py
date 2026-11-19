"""Check that Supabase has the schema the app expects.   python -m app.verify_db

Goes through the same REST/RPC path the app uses, so a PASS here means /upload, /search and
/chat have what they need. Exit code 1 if anything fails.
"""
import sys

from app import db
from app.config import get_settings


def check(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))
        return True
    except Exception as e:  # noqa: BLE001 - we want the message, whatever it is
        msg = getattr(e, "message", None) or str(e)
        print(f"FAIL  {name}  -> {str(msg)[:200]}")
        return False


def main() -> None:
    if get_settings().vector_store == "local":
        print(f"VECTOR_STORE=local: no database to check (documents live in {get_settings().local_store_path}).")
        sys.exit(0)
    sb = db.get_client()
    docs: list[dict] = []

    def documents_table():
        docs.extend(sb.table("documents").select("doc_id,filename,num_chunks,active_batch,scope,owner_id,file_hash").execute().data)
        return f"{len(docs)} document(s): {[d['doc_id'] for d in docs]}"

    def chunks_have_batch():
        r = sb.table("chunks").select("id", count="exact").is_("batch_id", "null").limit(1).execute()
        if r.count:
            raise RuntimeError(f"{r.count} chunks have no batch_id (re-run the migration)")
        return "every chunk has a batch_id"

    def active_chunks_match_documents():
        if not docs:
            raise RuntimeError("no documents registered yet (run the migration, or upload a PDF)")
        for d in docs:
            r = sb.table("chunks").select("id", count="exact").eq("doc_id", d["doc_id"]).eq("batch_id", d["active_batch"]).limit(1).execute()
            if not r.count:
                raise RuntimeError(f"{d['doc_id']}: no chunks in its active batch")
        return "active batches contain chunks"

    def match_function():
        rows = db.search_chunks([0.01] * 1536, 3)
        if docs and not rows:
            raise RuntimeError("match_documents returned nothing although documents exist")
        return f"returned {len(rows)} row(s)"

    def isolation():
        stranger = "00000000-0000-4000-8000-000000000000"
        leaked = db.search_chunks([0.01] * 1536, 5, owner_id=stranger, scope="mine")
        if leaked:
            raise RuntimeError("an owner with no documents received rows from scope 'mine'")
        user_ids = {d["doc_id"] for d in docs if d["scope"] == "user"}
        anon = {r["doc_id"] for r in db.search_chunks([0.01] * 1536, 20)}
        if anon & user_ids:
            raise RuntimeError("private documents are visible without an owner")
        return "no private data visible to strangers"

    def interactions_table():
        sb.table("interactions").select("request_id", count="exact").limit(1).execute()
        return "chat logging and feedback can be stored"

    results = [
        check("documents table exists", documents_table),
        check("chunks.batch_id populated", chunks_have_batch),
        check("documents point at real chunk batches", active_chunks_match_documents),
        check("match_documents() works with new signature", match_function),
        check("private documents are isolated", isolation),
        check("interactions table exists (003)", interactions_table),
    ]
    print("\nALL GOOD" if all(results) else
          "\nFIX THE FAILURES ABOVE (run db/migrations/001_init.sql, 002_scopes.sql and 003_interactions.sql)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
