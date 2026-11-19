"""Manage the shared (core) knowledge base.  Needs the service-role key in .env, i.e. only you.

    python -m app.admin add data/Human-Nutrition-2020.pdf --doc-id human-nutrition-2020 --title "Human Nutrition (2020)"
    python -m app.admin list
    python -m app.admin remove human-nutrition-2020
    python -m app.admin import-local              # copy core docs from data/local_store.pkl to Supabase

`add` on an existing --doc-id publishes a new version atomically (old one stays live until the
new one is fully stored), so keeping the knowledge base current is just re-running the command.

`import-local` reuses the embeddings already computed in local mode (VECTOR_STORE=local), so moving
the library into Supabase costs no OpenAI credits. Only core documents are copied: private uploads
belong to browser ids that mean nothing outside the machine they were made on.
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

from app import db
from app.config import get_settings
from app.ingest_service import Conflict, ingest_pdf
from app.pdf_check import PdfRejected

DOC_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,62}$")


def cmd_add(args: argparse.Namespace) -> int:
    if not DOC_ID.match(args.doc_id) or args.doc_id.startswith("u-"):
        print("doc-id must be 2-63 chars of a-z 0-9 _ . - and must not start with 'u-' (reserved for user uploads).")
        return 2
    path = Path(args.pdf)
    data = path.read_bytes()
    limit = get_settings().max_upload_mb * 1024 * 1024
    if len(data) > limit:
        print(f"{path.name} is {len(data) / 1e6:.0f} MB; limit is {get_settings().max_upload_mb} MB (MAX_UPLOAD_MB).")
        return 2
    try:
        res = ingest_pdf(data, args.title or path.name, doc_id=args.doc_id, scope="core")
    except (PdfRejected, Conflict) as e:
        print(f"REJECTED: {e}")
        return 1
    print(f"{res['status']}: {res['doc_id']} - {res['chunks_inserted']} chunks from {res['pages']} pages")
    for w in res.get("warnings", []):
        print(f"  warning: {w}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    docs = db.list_documents(None)["core"]
    if not docs:
        print("(no core documents yet)")
    for d in docs:
        print(f"{d['doc_id']:<28} {d['num_chunks']:>6} chunks  {d['updated_at'][:10]}  {d['filename']}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    doc = db.get_document(args.doc_id)
    if not doc or doc["scope"] != "core":
        print(f"No core document '{args.doc_id}'. (User documents are deleted by their owners.)")
        return 1
    db.delete_document(args.doc_id)
    print(f"removed {args.doc_id}")
    return 0


def cmd_import_local(args: argparse.Namespace) -> int:
    s = get_settings()
    if s.vector_store == "local":
        print("VECTOR_STORE=local: the target is the local store itself. Unset it to import into Supabase.")
        return 2
    path = Path(args.path or s.local_store_path)
    if not path.exists():
        print(f"{path} not found. Build it first with VECTOR_STORE=local python -m app.admin add ...")
        return 1
    with path.open("rb") as f:
        state = pickle.load(f)  # our own file, written by app/local_store.py
    core = [d for d in state.get("documents", {}).values() if d.get("scope") == "core"]
    if args.doc_id:
        core = [d for d in core if d["doc_id"] in args.doc_id]
    if not core:
        print("No core documents to import.")
        return 1
    for d in core:
        stored = state["chunks"].get(d["doc_id"])
        if not stored or not stored["rows"]:
            print(f"skip {d['doc_id']}: no chunks")
            continue
        rows = [
            {"chunk_index": r["chunk_index"], "content": r["content"], "metadata": r["metadata"],
             "embedding": [float(x) for x in vec]}
            for r, vec in zip(stored["rows"], stored["vectors"], strict=True)
        ]
        # Same atomic, versioned write as a normal upload: the old version stays live until this one is stored.
        n = db.replace_document(d["doc_id"], d.get("filename") or d["doc_id"], rows, scope="core", file_hash=d.get("file_hash"))
        print(f"imported {d['doc_id']}: {n} chunks")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m app.admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="add or update a core document")
    a.add_argument("pdf")
    a.add_argument("--doc-id", required=True)
    a.add_argument("--title")
    a.set_defaults(fn=cmd_add)
    sub.add_parser("list", help="list core documents").set_defaults(fn=cmd_list)
    r = sub.add_parser("remove", help="delete a core document")
    r.add_argument("doc_id")
    r.set_defaults(fn=cmd_remove)
    i = sub.add_parser("import-local", help="copy core documents (with their embeddings) from the local store to Supabase")
    i.add_argument("--path", help="local store file (default: LOCAL_STORE_PATH)")
    i.add_argument("--doc-id", action="append", help="only this document (repeatable)")
    i.set_defaults(fn=cmd_import_local)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
