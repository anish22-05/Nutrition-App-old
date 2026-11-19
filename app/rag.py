"""Retrieval + answering steps shared by the API and the eval harness.

Keeping this in one place means the thing we *measure* in `eval/` is the same code
path the API *serves* - otherwise an eval score tells you nothing about production.
"""
import logging
import re

from app import db, llm, trace
from app.config import get_settings
from app.retrieval import merge_ranked, rerank

log = logging.getLogger(__name__)
NO_ANSWER = llm.NO_ANSWER

# Only questions that look multi-part pay for the extra planning call.
_MULTI = re.compile(r"\b(?:and|vs\.?|versus|compare[ds]?|comparison|difference|differ|between|both|or)\b|,", re.IGNORECASE)


def looks_multipart(question: str) -> bool:
    return bool(_MULTI.search(question))


def retrieve(
    query: str,
    top_k: int,
    doc_id: str | None = None,
    min_similarity: float = 0.0,
    owner_id: str | None = None,
    scope: str = "all",
    *,
    query_vec: list[float] | None = None,
) -> list[dict]:
    s = get_settings()
    if query_vec is None:
        with trace.span("embed"):
            query_vec = llm.embed_texts([query])[0]
    with trace.span("search"):
        if not s.rerank_enabled:
            return db.search_chunks(query_vec, top_k, doc_id, min_similarity, owner_id, scope)
        # Two-stage: cheap vector search for a wider candidate list, then a cross-encoder re-orders it.
        # The similarity gate still uses the cosine score from stage one, so refusals behave the same.
        cands = db.search_chunks(query_vec, max(top_k, s.rerank_candidates), doc_id, min_similarity, owner_id, scope)
    if len(cands) <= 1:
        return cands[:top_k]
    with trace.span("rerank"):
        scores = rerank(query, [c["content"] for c in cands], s.reranker_model)
    best = sorted(zip(scores, range(len(cands)), strict=True), key=lambda x: (-x[0], x[1]))
    return [cands[i] for _, i in best[:top_k]]


def retrieve_many(
    queries: list[str],
    vecs: list[list[float]],
    top_k: int,
    doc_id: str | None = None,
    min_similarity: float = 0.0,
    owner_id: str | None = None,
    scope: str = "all",
) -> list[dict]:
    """One search per sub-query, merged round-robin so each part of the question gets evidence."""
    if len(queries) == 1:
        return retrieve(queries[0], top_k, doc_id, min_similarity, owner_id, scope, query_vec=vecs[0])
    lists = [retrieve(q, top_k, doc_id, min_similarity, owner_id, scope, query_vec=v) for q, v in zip(queries, vecs, strict=True)]
    return merge_ranked(lists, top_k)


def plan_queries(question: str) -> list[str]:
    """Sub-queries for a multi-part question; a failure must never break chat."""
    s = get_settings()
    if not s.query_decomposition:
        return [question]
    try:
        with trace.span("plan"):
            return llm.plan_queries(question, s.max_subqueries)
    except Exception:
        log.warning("query planning failed", exc_info=True)
        return [question]


def answer(question: str, matches: list[dict], *, use_tools: bool = False, safety_note: str | None = None) -> str:
    """No usable context (and no tools) -> refuse without spending an LLM call."""
    return answer_with_tools(question, matches, use_tools=use_tools, safety_note=safety_note)[0]


def answer_with_tools(
    question: str, matches: list[dict], *, use_tools: bool = False, safety_note: str | None = None
) -> tuple[str, list[dict]]:
    if not matches and not use_tools:
        return NO_ANSWER, []
    with trace.span("generate"):
        return llm.generate_answer(question, matches, use_tools=use_tools, safety_note=safety_note)


def standalone_question(history: list[dict], question: str) -> str:
    """Follow-ups become self-contained queries; a failure here must never break chat."""
    try:
        with trace.span("condense"):
            return llm.condense_question(history, question)
    except Exception:
        log.warning("question rewrite failed", exc_info=True)
        return question


def stream_answer(question: str, matches: list[dict], *, use_tools: bool = False, safety_note: str | None = None):
    """Streaming twin of answer_with_tools: yields llm.answer_events() events."""
    return llm.answer_events(question, matches, use_tools=use_tools, safety_note=safety_note)
