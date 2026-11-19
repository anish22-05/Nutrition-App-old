"""Retrieval building blocks beyond plain vector search. Pure functions plus one lazy model.

* BM25     - classic keyword ranking. Embeddings blur exact tokens ("DFE", "B12", "DEXA", numbers);
             BM25 nails them. The two fail on different queries, which is why they are fused.
* rrf      - Reciprocal Rank Fusion. Combines ranked lists using only *ranks*, so no need to
             normalise cosine scores against BM25 scores (they live on different scales).
* rerank   - a cross-encoder reads (query, passage) together instead of comparing two
             independent vectors. Much more precise, far too slow to run on the whole corpus,
             so it only re-orders a short candidate list from the cheap first stage.
"""
import math
import re
from collections import Counter
from functools import lru_cache

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "they", "this", "to", "was", "were", "what", "when", "which", "who", "why", "will", "with"]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 over a fixed list of texts (k1/b are the usual defaults)."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(texts)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # inverted index: term -> [(doc, tf)], so a query only touches documents containing its terms
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self.tfs):
            for t, c in tf.items():
                self.postings.setdefault(t, []).append((i, c))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in self.postings[t]:
                norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg_len or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * c * (self.k1 + 1) / (c + norm)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse several best-first id lists. score(d) = sum 1/(k + rank). k=60 is the standard
    constant from the original paper; it damps the advantage of being ranked #1 vs #3."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])


DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_reranker(model: str):
    from fastembed.rerank.cross_encoder import (
        TextCrossEncoder,  # heavy import, only when used
    )

    return TextCrossEncoder(model_name=model)


def rerank(query: str, texts: list[str], model: str = DEFAULT_RERANKER) -> list[float]:
    """Relevance score per text (higher = better). Order of `texts` is preserved."""
    if not texts:
        return []
    return [float(s) for s in _load_reranker(model).rerank(query, texts)]


def merge_ranked(lists: list[list[dict]], k: int) -> list[dict]:
    """Interleave several ranked result lists (one per sub-query), dropping duplicates.

    Round-robin rather than sorting by score: each sub-query gets its best passage in first, so
    "compare A and B" ends up with evidence for both A and B instead of five passages about A.
    """
    seen: set = set()
    out: list[dict] = []
    for rank in range(max((len(x) for x in lists), default=0)):
        for results in lists:
            if rank < len(results):
                r = results[rank]
                key = (r.get("doc_id"), r.get("chunk_index"), r.get("content") if r.get("chunk_index") is None else None)
                if key not in seen:
                    seen.add(key)
                    out.append(r)
                    if len(out) == k:
                        return out
    return out
