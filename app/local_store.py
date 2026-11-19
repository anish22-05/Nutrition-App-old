"""File-backed stand-in for Supabase (VECTOR_STORE=local): run the whole app with no database.

Same functions and the same semantics as app/db.py, including the parts that matter for
correctness: versioned (atomic) document replacement, core vs private scopes with owner
isolation inside search, and the similarity cut-off. Search is exact brute-force cosine, which
is instant at this scale (a few thousand chunks).

The state is one pickle file written atomically (temp file + rename), so the admin CLI (one
process) and the API (another) share it: the API reloads when the file's mtime changes.
Not for multi-instance production: there is no cross-process locking beyond the atomic rename.
"""
import logging
import os
import pickle
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.RLock()
_state: dict | None = None
_mtime: float | None = None


def _path() -> Path:
    return Path(get_settings().local_store_path)


def _empty() -> dict:
    return {"documents": {}, "chunks": {}, "interactions": {}}


def _load() -> dict:
    """Current state, re-read if another process changed the file."""
    global _state, _mtime
    p = _path()
    mtime = p.stat().st_mtime if p.exists() else None
    if _state is None or mtime != _mtime:
        if mtime is None:
            _state = _empty()
        else:
            with p.open("rb") as f:
                _state = pickle.load(f)
            for k, v in _empty().items():
                _state.setdefault(k, v)
        _mtime = mtime
    return _state


def _save() -> None:
    global _mtime
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        pickle.dump(_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    _mtime = p.stat().st_mtime


def _public(doc: dict, cols: tuple[str, ...]) -> dict:
    return {c: doc.get(c) for c in cols}


DOC_COLS = ("doc_id", "filename", "num_chunks", "scope", "owner_id", "updated_at")


def replace_document(doc_id, filename, rows, *, scope="core", owner_id=None, file_hash=None) -> int:
    vectors = np.asarray([r["embedding"] for r in rows], dtype=np.float32).reshape(len(rows), -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) if len(rows) else np.ones((0, 1))
    with _lock:
        st = _load()
        st["chunks"][doc_id] = {
            "batch_id": str(uuid.uuid4()),
            "rows": [{"chunk_index": r["chunk_index"], "content": r["content"], "metadata": r["metadata"]} for r in rows],
            "vectors": vectors / np.clip(norms, 1e-12, None),
        }
        st["documents"][doc_id] = {
            "doc_id": doc_id, "filename": filename, "num_chunks": len(rows), "scope": scope,
            "owner_id": owner_id, "file_hash": file_hash, "updated_at": datetime.now(UTC).isoformat(),
        }
        _save()
    return len(rows)


def get_document(doc_id):
    with _lock:
        d = _load()["documents"].get(doc_id)
        return _public(d, (*DOC_COLS, "file_hash")) if d else None


def find_document_by_hash(file_hash, scope, owner_id):
    with _lock:
        for d in _load()["documents"].values():
            if d.get("file_hash") == file_hash and d["scope"] == scope and d.get("owner_id") == owner_id:
                return _public(d, DOC_COLS)
    return None


def list_documents(owner_id):
    cols = ("doc_id", "filename", "num_chunks", "updated_at")
    with _lock:
        docs = list(_load()["documents"].values())
    core = sorted((_public(d, cols) for d in docs if d["scope"] == "core"), key=lambda d: d["doc_id"])
    mine = []
    if owner_id:
        mine = sorted((_public(d, cols) for d in docs if d["scope"] == "user" and d.get("owner_id") == owner_id),
                      key=lambda d: d["updated_at"], reverse=True)
    return {"core": core, "mine": mine}


def count_user_documents(owner_id, excluding_doc_id=None):
    with _lock:
        return sum(1 for d in _load()["documents"].values()
                   if d["scope"] == "user" and d.get("owner_id") == owner_id and d["doc_id"] != excluding_doc_id)


def delete_document(doc_id):
    with _lock:
        st = _load()
        st["documents"].pop(doc_id, None)
        st["chunks"].pop(doc_id, None)
        _save()


def search_chunks(query_embedding, top_k, doc_id=None, min_similarity=0.0, owner_id=None, scope="all"):
    """Mirrors SQL match_documents(): core hits + (only with an owner) that owner's hits."""
    q = np.asarray(query_embedding, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    with _lock:
        st = _load()
        docs = list(st["documents"].values())
        chunks = st["chunks"]

    def hits(selected: list[dict]) -> list[dict]:
        out = []
        for d in selected:
            c = chunks.get(d["doc_id"])
            if not c or not len(c["rows"]):
                continue
            sims = c["vectors"] @ q
            for i in np.argsort(-sims)[:top_k]:
                r = c["rows"][int(i)]
                out.append({"id": f"{d['doc_id']}:{r['chunk_index']}", "doc_id": d["doc_id"], "chunk_index": r["chunk_index"],
                            "content": r["content"], "metadata": r["metadata"], "similarity": float(sims[int(i)])})
        return sorted(out, key=lambda h: -h["similarity"])[:top_k]

    pick = (lambda d: doc_id is None or d["doc_id"] == doc_id)
    found = []
    if scope in ("all", "core"):
        found += hits([d for d in docs if d["scope"] == "core" and pick(d)])
    if scope in ("all", "mine") and owner_id:
        found += hits([d for d in docs if d["scope"] == "user" and d.get("owner_id") == owner_id and pick(d)])
    found = [h for h in found if h["similarity"] >= min_similarity]
    return sorted(found, key=lambda h: -h["similarity"])[:top_k]


# --- interactions (questions, answers, feedback) -------------------------------------------------

def log_interaction(row: dict) -> None:
    with _lock:
        st = _load()
        st["interactions"][row["request_id"]] = {"created_at": datetime.now(UTC).isoformat(), "rating": None, "comment": None, **row}
        _save()


def rate_interaction(request_id, owner_id, rating, comment) -> bool:
    with _lock:
        st = _load()
        it = st["interactions"].get(request_id)
        if not it or it.get("owner_id") != owner_id:
            return False
        it.update(rating=rating, comment=comment, rated_at=datetime.now(UTC).isoformat())
        _save()
        return True


def list_interactions(rated_only=True, limit=500) -> list[dict]:
    with _lock:
        rows = list(_load()["interactions"].values())
    if rated_only:
        rows = [r for r in rows if r.get("rating") is not None]
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)[:limit]
