"""One ingestion pipeline for every upload path (user API, admin CLI): validate -> dedupe ->
chunk -> embed -> store. Embedding is the expensive step, so every cheap rejection happens first."""
import hashlib
import logging
from collections.abc import Callable

from app import db, guard, llm, pdf_check
from app.config import get_settings
from app.ingestion import build_chunks_from_pdf, embedding_text

log = logging.getLogger(__name__)


class QuotaExceeded(Exception):
    """User hit a per-owner limit. Message is shown to them."""


class Conflict(Exception):
    """The target doc_id already belongs to someone else / another scope."""


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    *,
    doc_id: str,
    scope: str,
    owner_id: str | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> dict:
    """`progress(stage, fraction)` is called as work advances (used by background upload jobs)."""
    s = get_settings()
    report = progress or (lambda _stage, _frac: None)
    report("checking", 0.02)
    is_user = scope == "user"
    if is_user and not owner_id:
        raise ValueError("user documents need an owner_id")

    # 1. cheap structural checks
    info = pdf_check.inspect_pdf(file_bytes, s.user_max_pages if is_user else s.admin_max_pages)

    # 2. never let one doc_id change hands or flip between core and private
    existing = db.get_document(doc_id)
    if existing and (existing["scope"] != scope or existing.get("owner_id") != owner_id):
        raise Conflict("That document id is already taken.")

    # 3. same bytes already stored in this space -> nothing to pay for
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    dup = db.find_document_by_hash(file_hash, scope, owner_id)
    if dup:
        return {
            "status": "duplicate",
            "doc_id": dup["doc_id"],
            "title": dup["filename"],
            "chunks_inserted": 0,
            "pages": info.pages,
            "warnings": [f"This exact file is already uploaded as '{dup['filename']}'; nothing was changed."],
        }

    # 4. per-user quota (replacing your own document does not count against it)
    if is_user and db.count_user_documents(owner_id, excluding_doc_id=doc_id) >= s.user_max_docs:
        raise QuotaExceeded(f"You can keep up to {s.user_max_docs} documents. Delete one to upload another.")

    # 5. extract + judge the text
    report("reading text", 0.1)
    try:
        chunks, metas = build_chunks_from_pdf(
            file_bytes,
            filename,
            sents_per_chunk=s.sents_per_chunk,
            overlap=s.sent_overlap,
            max_tokens=s.max_chunk_tokens,
            min_tokens=s.min_chunk_tokens,
        )
    except Exception as e:
        log.exception("PDF text extraction failed")
        raise pdf_check.PdfRejected("The text of this PDF could not be read.") from e

    pages_with_text, warnings = pdf_check.assess_text(
        info,
        chunks,
        metas,
        min_text_page_ratio=s.min_text_page_ratio,
        max_garble_ratio=s.max_garble_ratio,
    )
    if is_user and len(chunks) > s.user_max_chunks:
        raise pdf_check.PdfRejected(
            f"This document is too large ({len(chunks)} sections; the limit is {s.user_max_chunks}). Upload a shorter one."
        )

    # Untrusted text addressed to an AI is kept (it may be legitimate) but neutralised at answer
    # time (app/guard.py); tell the uploader so it isn't a silent surprise.
    suspicious = guard.count_suspicious(chunks)
    if suspicious:
        warnings.append(
            f"{suspicious} section(s) contain text that reads like instructions to an AI. "
            "They are kept as plain document text and can't change how the assistant behaves."
        )

    # 6. only now spend money: embed (with section context), then store atomically
    report("embedding", 0.25)
    texts = [embedding_text(c, m, s.contextual_embeddings) for c, m in zip(chunks, metas, strict=True)]
    vectors = []
    step = max(1, s.embed_batch_size)
    for i in range(0, len(texts), step):
        vectors += llm.embed_texts(texts[i : i + step])
        report("embedding", 0.25 + 0.6 * min(1.0, (i + step) / len(texts)))
    report("storing", 0.9)
    rows = [
        {"chunk_index": i, "content": c, "metadata": m, "embedding": v}
        for i, (c, m, v) in enumerate(zip(chunks, metas, vectors, strict=True))
    ]
    n = db.replace_document(doc_id, filename, rows, scope=scope, owner_id=owner_id, file_hash=file_hash)
    log.info("ingested doc_id=%s scope=%s chunks=%d", doc_id, scope, n)
    report("done", 1.0)
    return {
        "status": "success",
        "doc_id": doc_id,
        "title": filename,
        "chunks_inserted": n,
        "pages": info.pages,
        "pages_with_text": pages_with_text,
        "warnings": warnings,
    }
