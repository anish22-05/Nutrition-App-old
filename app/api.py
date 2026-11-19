import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import db, llm, rag, safety, tools, trace, usage
from app.auth import resolve_owner
from app.cache import SemanticCache
from app.config import get_settings
from app.ingest_service import Conflict, QuotaExceeded, ingest_pdf
from app.jobs import JobStore
from app.pdf_check import PdfRejected
from app.ratelimit import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

settings = get_settings()
app = FastAPI(title="Nutrition RAG API")
limiter = RateLimiter()
cache = SemanticCache(settings.cache_similarity, settings.cache_ttl_s, settings.cache_max_entries)
jobs = JobStore()

# path -> requests per minute. Only the endpoints that cost money (or write) are limited.
LIMITED = {
    "/upload": settings.rate_limit_upload_per_min,
    "/upload/async": settings.rate_limit_upload_per_min,
    "/chat": settings.rate_limit_chat_per_min,
    "/chat/stream": settings.rate_limit_chat_per_min,
    "/feedback": 30,
}


@app.middleware("http")
async def guard_and_log(request: Request, call_next):
    """Rate limit the expensive endpoints, and log method/path/status/latency/cost/steps per request."""
    t0 = time.perf_counter()
    used = usage.start()
    trace.start()
    limit = LIMITED.get(request.url.path)
    if limit and request.method == "POST":
        ip = request.client.host if request.client else "unknown"
        owner = request.headers.get("x-owner-id", "-")
        for key in (f"ip:{ip}:{request.url.path}", f"owner:{owner}:{request.url.path}"):
            if not limiter.allow(key, limit, 60):
                log.warning("rate limited %s %s", key, request.url.path)
                return JSONResponse({"detail": "Too many requests; please slow down."}, status_code=429, headers={"Retry-After": "60"})
    response = await call_next(request)
    if request.url.path != "/chat/stream":  # streamed answers are logged when the stream ends
        cost = f" | {usage.summary()}" if used.calls else ""
        steps = trace.summary()
        log.info("%s %s -> %s in %.0f ms%s%s", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - t0) * 1000, cost, f" | steps {steps}" if steps else "")
    return response


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)
    doc_id: str | None = None
    scope: Literal["all", "core", "mine"] = "all"  # core = shared knowledge base, mine = your uploads


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(SearchRequest):
    history: list[Turn] = Field(default_factory=list, max_length=10)  # earlier turns, oldest first


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=8, max_length=64)
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1000)


# NOTE: these are plain `def` (not `async def`) on purpose. The OpenAI / Supabase / PyMuPDF
# calls are blocking; FastAPI runs sync endpoints in a threadpool, so one slow request
# no longer freezes the whole event loop.
#
# Identity: without login each browser holds a random UUID and sends it as `X-Owner-Id` (an
# unguessable private link). With AUTH_REQUIRED=true that header is ignored and only a verified
# Supabase token identifies a user (app/auth.py).


def current_owner(
    x_owner_id: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    return resolve_owner(x_owner_id, authorization)


Owner = Annotated[str | None, Depends(current_owner)]


def user_doc_id(owner_id: str, filename: str) -> str:
    """Server-chosen, so a user can never pick (and overwrite) a core document's id."""
    slug = re.sub(r"[^a-z0-9]+", "-", PurePosixPath(filename).stem.lower()).strip("-")[:40] or "document"
    return "u-" + hashlib.sha256(f"{owner_id}:{slug}".encode()).hexdigest()[:20]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store": settings.vector_store,
        "features": {
            "tools": settings.tools_enabled,
            "query_decomposition": settings.query_decomposition,
            "contextual_embeddings": settings.contextual_embeddings,
            "rerank": settings.rerank_enabled,
            "cache": settings.cache_enabled,
        },
        "cache": {"hits": cache.hits, "misses": cache.misses},
    }


# --- uploads ---------------------------------------------------------------------------------------

def _read_upload(file: UploadFile, owner_id: str | None) -> tuple[bytes, str]:
    if owner_id is None:
        if settings.auth_required:
            raise HTTPException(401, "Sign in to upload documents.")
        raise HTTPException(400, "X-Owner-Id header is required to upload.")
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    limit = settings.user_max_upload_mb * 1024 * 1024
    file_bytes = file.file.read(limit + 1)
    if len(file_bytes) > limit:
        raise HTTPException(413, f"File too large (max {settings.user_max_upload_mb} MB).")
    return file_bytes, filename


def _user_error(e: Exception) -> str:
    """Message safe to show a user for an ingestion failure (internal errors are not leaked)."""
    if isinstance(e, (PdfRejected, QuotaExceeded, Conflict)):
        return str(e)
    log.error("ingestion failed", exc_info=e)
    return "Ingestion failed; any previous version of this document is untouched."


@app.post("/upload")
def upload(file: Annotated[UploadFile, File()], owner_id: Owner):
    """Upload a PDF to the caller's private space and wait for it. (Core documents: `python -m app.admin`.)"""
    file_bytes, filename = _read_upload(file, owner_id)
    try:
        result = ingest_pdf(file_bytes, filename, doc_id=user_doc_id(owner_id, filename), scope="user", owner_id=owner_id)
    except PdfRejected as e:
        raise HTTPException(422, str(e)) from None
    except (QuotaExceeded, Conflict) as e:
        raise HTTPException(409, str(e)) from None
    except Exception:
        log.exception("ingestion failed")
        raise HTTPException(502, "Ingestion failed; any previous version of this document is untouched.") from None
    cache.invalidate_owner(owner_id)
    return result


@app.post("/upload/async", status_code=202)
def upload_async(file: Annotated[UploadFile, File()], owner_id: Owner):
    """Same as /upload, but returns a job id at once; poll GET /jobs/{job_id} for progress."""
    file_bytes, filename = _read_upload(file, owner_id)

    def work(progress):
        result = ingest_pdf(file_bytes, filename, doc_id=user_doc_id(owner_id, filename), scope="user",
                            owner_id=owner_id, progress=progress)
        cache.invalidate_owner(owner_id)
        return result

    return {"job_id": jobs.submit(owner_id, work, _user_error)}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, owner_id: Owner):
    job = jobs.get(job_id, owner_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


@app.get("/documents")
def documents(owner_id: Owner):
    """The shared knowledge base plus the caller's own documents."""
    return db.list_documents(owner_id)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, owner_id: Owner):
    doc = db.get_document(doc_id)
    # Same 404 for "missing" and "not yours" so ids of other people's documents can't be probed.
    if not doc or doc["scope"] != "user" or owner_id is None or doc.get("owner_id") != owner_id:
        raise HTTPException(404, "Document not found.")
    db.delete_document(doc_id)
    cache.invalidate_owner(owner_id)
    return {"status": "deleted", "doc_id": doc_id}


# --- search & chat -----------------------------------------------------------------------------------

def _retrieve(req: SearchRequest, owner_id: str | None, min_similarity: float) -> list[dict]:
    try:
        return rag.retrieve(req.query, req.top_k, req.doc_id, min_similarity, owner_id, req.scope)
    except Exception:
        log.exception("retrieval failed")
        raise HTTPException(502, "Retrieval backend error.") from None


@app.post("/search")
def search(req: SearchRequest, owner_id: Owner):
    return _retrieve(req, owner_id, min_similarity=0.0)


@dataclass
class Prepared:
    request_id: str
    question: str
    standalone: str
    safety: safety.SafetyCheck
    queries: list[str] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    use_tools: bool = False
    cached: dict | None = None
    cache_key: tuple | None = None
    vec: list[float] | None = None


def _prepare_chat(req: ChatRequest, owner_id: str | None) -> Prepared:
    """Rewrite the follow-up, check safety and the cache, split multi-part questions, retrieve."""
    first = safety.classify(req.query)
    if first.blocks_answer:  # emergency: answer at once, before any model call
        return Prepared(uuid.uuid4().hex, req.query, req.query, first, queries=[req.query])
    history = [t.model_dump() for t in req.history[-6:]]
    standalone = rag.standalone_question(history, req.query)
    check = safety.classify(req.query, standalone)
    p = Prepared(uuid.uuid4().hex, req.query, standalone, check, queries=[standalone])
    if check.blocks_answer:
        return p  # emergency only visible after resolving the follow-up: still no search or generation

    try:
        with trace.span("embed"):
            p.vec = llm.embed_texts([standalone])[0]
        private_visible = req.scope != "core"
        p.cache_key = (req.scope, req.doc_id, req.top_k, settings.tools_enabled, check.category,
                       owner_id if private_visible else None)
        if settings.cache_enabled:
            p.cached = cache.get(p.cache_key, p.vec)
            if p.cached:
                return p

        vecs = [p.vec]
        if rag.looks_multipart(standalone):
            p.queries = rag.plan_queries(standalone)
            if p.queries != [standalone]:
                with trace.span("embed"):
                    vecs = llm.embed_texts(p.queries)
        p.matches = rag.retrieve_many(p.queries, vecs, req.top_k, req.doc_id, settings.min_similarity, owner_id, req.scope)
    except Exception:
        log.exception("retrieval failed")
        raise HTTPException(502, "Retrieval backend error.") from None
    # With nothing retrieved, only pay for an LLM call if a tool could plausibly answer.
    p.use_tools = settings.tools_enabled and (bool(p.matches) or tools.might_need_tools(standalone))
    return p


def _finish(p: Prepared, owner_id: str | None, answer: str, tool_calls: list[dict], t0: float, *, store: bool) -> None:
    """Cache the answer and log the interaction. Best effort: never fails the request."""
    if store and settings.cache_enabled and p.cache_key and p.vec is not None and not p.cached:
        cache.put(p.cache_key, p.vec, {"answer": answer, "chunks": p.matches, "tools": tool_calls, "queries": p.queries},
                  owner_id if p.cache_key[-1] else None)
    if not settings.log_interactions:
        return
    try:
        db.log_interaction({
            "request_id": p.request_id,
            "owner_id": owner_id,
            "question": p.question,
            "standalone": p.standalone,
            "answer": answer,
            "sources": [{"doc_id": m.get("doc_id"), "chunk_index": m.get("chunk_index"),
                         "page": (m.get("metadata") or {}).get("page"), "similarity": m.get("similarity")} for m in p.matches],
            "tools": [{"name": t["name"], "arguments": t["arguments"]} for t in tool_calls],
            "safety": p.safety.category,
            "cached": bool(p.cached),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "cost_usd": usage.current_cost(),
        })
    except Exception:
        log.warning("could not log interaction (is db/migrations/003_interactions.sql applied?)", exc_info=True)


def _meta(p: Prepared) -> dict:
    return {"request_id": p.request_id, "standalone_question": p.standalone, "queries": p.queries,
            "safety": p.safety.to_dict(), "cached": bool(p.cached)}


@app.post("/chat")
def chat(req: ChatRequest, owner_id: Owner):
    t0 = time.perf_counter()
    p = _prepare_chat(req, owner_id)
    if p.safety.blocks_answer:
        answer, tool_calls = p.safety.notice, []
    elif p.cached:
        answer, tool_calls, p.matches, p.queries = p.cached["answer"], p.cached["tools"], p.cached["chunks"], p.cached["queries"]
    else:
        try:
            answer, tool_calls = rag.answer_with_tools(p.standalone, p.matches, use_tools=p.use_tools,
                                                       safety_note=p.safety.prompt_addition)
        except Exception:
            log.exception("generation failed")
            raise HTTPException(502, "LLM backend error.") from None
    _finish(p, owner_id, answer, tool_calls, t0, store=not p.safety.blocks_answer and answer != llm.WITHHELD)
    return {"answer": answer, "chunks": p.matches, "tools": tool_calls, "trace": trace.summary(), **_meta(p)}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, owner_id: Owner):
    """Newline-delimited JSON events: `sources`, then `tool` / `token` events, then `done` (or `error`).
    Sources arrive first so the UI can show them while the answer is still being written."""
    t0 = time.perf_counter()
    p = _prepare_chat(req, owner_id)  # errors here are still proper HTTP errors

    def events():
        if p.cached:
            p.matches, p.queries = p.cached["chunks"], p.cached["queries"]
        yield json.dumps({"type": "sources", "chunks": p.matches, **_meta(p)}) + "\n"
        parts: list[str] = []
        tool_calls: list[dict] = []
        ok = False
        try:
            if p.safety.blocks_answer:
                parts.append(p.safety.notice)
                yield json.dumps({"type": "token", "text": p.safety.notice}) + "\n"
            elif p.cached:
                tool_calls = p.cached["tools"]
                for t in tool_calls:
                    yield json.dumps({"type": "tool", **t}) + "\n"
                parts.append(p.cached["answer"])
                yield json.dumps({"type": "token", "text": p.cached["answer"]}) + "\n"
            elif not p.matches and not p.use_tools:
                parts.append(rag.NO_ANSWER)
                yield json.dumps({"type": "token", "text": rag.NO_ANSWER}) + "\n"
            else:
                with trace.span("generate"):
                    for e in rag.stream_answer(p.standalone, p.matches, use_tools=p.use_tools, safety_note=p.safety.prompt_addition):
                        if e["type"] == "withheld":
                            parts = [llm.WITHHELD]
                            yield json.dumps({"type": "withheld", "text": llm.WITHHELD}) + "\n"
                            break
                        if e["type"] == "tool":
                            tool_calls.append({k: e[k] for k in ("id", "name", "arguments", "result")})
                        else:
                            parts.append(e["text"])
                        yield json.dumps(e) + "\n"
            ok = True
            yield json.dumps({"type": "done", "trace": trace.summary()}) + "\n"
        except Exception:
            log.exception("streaming generation failed")
            yield json.dumps({"type": "error", "message": "LLM backend error."}) + "\n"
        finally:
            answer = "".join(parts)
            if ok:
                _finish(p, owner_id, answer, tool_calls, t0, store=not p.safety.blocks_answer and answer != llm.WITHHELD)
            log.info("POST /chat/stream finished in %.0f ms | %s | steps %s", (time.perf_counter() - t0) * 1000,
                     usage.summary() or "no OpenAI calls", trace.summary())

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/feedback")
def feedback(req: FeedbackRequest, owner_id: Owner):
    """Thumbs up/down on an answer. Only the person who asked can rate it."""
    try:
        ok = db.rate_interaction(req.request_id, owner_id, req.rating, req.comment)
    except Exception:
        log.exception("feedback failed")
        raise HTTPException(503, "Feedback could not be saved right now.") from None
    if not ok:
        raise HTTPException(404, "Answer not found.")
    return {"status": "saved"}
