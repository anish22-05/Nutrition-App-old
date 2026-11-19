"""An in-memory vector index built straight from a PDF.

Why not just query Supabase? Evaluation needs to be (a) reproducible, (b) runnable without
touching production data, and (c) able to re-chunk/re-embed under different settings to
compare strategies. A brute-force cosine search over a few thousand vectors is exact and
instant, so nothing is lost versus pgvector's approximate HNSW at this scale.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

from app import llm
from app.config import get_settings
from app.ingestion import build_chunks_from_pdf, embedding_text
from app.retrieval import BM25, merge_ranked, rerank, rrf

pymupdf.TOOLS.mupdf_display_errors(False)  # some PDFs embed corrupt images; the warnings are noise
CACHE_DIR = Path(__file__).parent / ".cache"


@dataclass
class Chunk:
    index: int
    text: str
    page: int
    source: str
    section: str | None = None


class InMemoryIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        self.chunks = chunks
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.clip(norms, 1e-12, None)  # unit length -> dot product == cosine
        self._bm25: BM25 | None = None

    def _row(self, i: int, sim: float) -> dict:
        c = self.chunks[i]
        return {
            "doc_id": c.source,
            "chunk_index": c.index,
            "content": c.text,
            "metadata": {"page": c.page, "source": c.source, **({"section": c.section} if c.section else {})},
            "similarity": float(sim),  # always the cosine, whatever ranked it: the similarity gate needs it
        }

    def search(
        self,
        query_vec,
        k: int,
        min_similarity: float = 0.0,
        *,
        query_text: str | None = None,
        mode: str = "vector",
        candidates: int = 30,
        reranker: str | None = None,
    ) -> list[dict]:
        """Same row shape as the SQL `match_documents` function, so downstream code is shared.

        mode: "vector" | "bm25" | "hybrid" (vector + BM25 fused with RRF).
        reranker: cross-encoder model name; re-orders the top `candidates` of the first stage.
        """
        q = np.asarray(query_vec, dtype=float)
        q = q / max(np.linalg.norm(q), 1e-12)
        sims = self.vectors @ q
        pool = max(k, candidates) if reranker else k

        vec_rank = [int(i) for i in np.argsort(-sims)[: max(pool, candidates)]]
        if mode == "vector":
            order = vec_rank
        else:
            if query_text is None:
                raise ValueError("bm25/hybrid search needs query_text")
            if self._bm25 is None:
                self._bm25 = BM25([c.text for c in self.chunks])
            kw_rank = [i for i, _ in self._bm25.search(query_text, max(pool, candidates))]
            order = kw_rank if mode == "bm25" else rrf([vec_rank, kw_rank])
        order = order[:pool]

        if reranker:
            if query_text is None:
                raise ValueError("reranking needs query_text")
            scores = rerank(query_text, [self.chunks[i].text for i in order], reranker)
            order = [i for _, i in sorted(zip(scores, order, strict=True), key=lambda x: -x[0])]

        return [self._row(i, sims[i]) for i in order[:k] if sims[i] >= min_similarity]


def _chunking_params() -> dict:
    s = get_settings()
    return {
        "sents_per_chunk": s.sents_per_chunk,
        "overlap": s.sent_overlap,
        "max_tokens": s.max_chunk_tokens,
        "min_tokens": s.min_chunk_tokens,
    }


def load_chunks(pdf_path: Path) -> list[Chunk]:
    texts, metas = build_chunks_from_pdf(pdf_path.read_bytes(), pdf_path.name, **_chunking_params())
    return [Chunk(i, t, m["page"], m["source"], m.get("section")) for i, (t, m) in enumerate(zip(texts, metas, strict=True))]


def build_index(pdf_path: Path, use_cache: bool = True, contextual: bool | None = None) -> InMemoryIndex:
    """Chunk the PDF with the app's own settings and embed it (with section context, like the app,
    unless contextual=False). Vectors are cached on disk, keyed by (file bytes, chunking params,
    embed model, contextual), so re-runs cost nothing."""
    if contextual is None:
        contextual = get_settings().contextual_embeddings
    chunks = load_chunks(pdf_path)
    if not chunks:
        raise SystemExit(f"No text could be extracted from {pdf_path.name} (corrupt or scanned PDF?).")

    key = hashlib.sha256(
        pdf_path.read_bytes()
        + json.dumps([_chunking_params(), get_settings().embed_model] + (["contextual-v1"] if contextual else [])).encode()
    ).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.npy"

    if use_cache and cache_file.exists():
        vectors = np.load(cache_file)
        if len(vectors) == len(chunks):
            return InMemoryIndex(chunks, vectors)

    texts = [embedding_text(c.text, {"section": c.section}, contextual) for c in chunks]
    vectors = np.asarray(llm.embed_texts(texts), dtype=float)
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache_file, vectors)
    return InMemoryIndex(chunks, vectors)


def search_planned(index: InMemoryIndex, queries: list[str], vecs: list, k: int, min_similarity: float = 0.0, **kw) -> list[dict]:
    """Eval twin of rag.retrieve_many: one search per sub-query, merged round-robin."""
    if len(queries) == 1:
        return index.search(vecs[0], k, min_similarity, query_text=queries[0], **kw)
    lists = [index.search(v, k, min_similarity, query_text=q, **kw) for q, v in zip(queries, vecs, strict=True)]
    return merge_ranked(lists, k)


def plan_all(questions: list[str], workers: int = 8) -> dict[str, list[str]]:
    """Sub-queries for every question that looks multi-part (same gate as the API), cached on disk."""
    from concurrent.futures import ThreadPoolExecutor

    from app import rag

    path = CACHE_DIR / f"plans-{get_settings().chat_model}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [q for q in questions if rag.looks_multipart(q) and q not in cache]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for q, plan in zip(todo, pool.map(lambda q: llm.plan_queries(q, get_settings().max_subqueries), todo), strict=True):
            cache[q] = plan
    if todo:
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return {q: (cache.get(q) if rag.looks_multipart(q) else None) or [q] for q in questions}
