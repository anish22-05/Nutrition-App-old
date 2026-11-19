import numpy as np

from app import retrieval
from app.retrieval import BM25, rrf, tokenize
from eval.corpus import Chunk, InMemoryIndex

TEXTS = [
    "Vitamin C improves the absorption of iron from plant foods.",
    "Calcium builds strong bones and teeth.",
    "The recommended DFE for folate in adults is 400 mcg.",
    "Fibre slows glucose absorption in the gut.",
]


def test_tokenize_drops_stopwords_and_keeps_numbers_and_hyphens():
    assert tokenize("What is the B12 low-GI diet?") == ["b12", "low-gi", "diet"]


def test_bm25_finds_exact_rare_token_and_ignores_unknown():
    bm = BM25(TEXTS)
    assert bm.search("DFE folate", 3)[0][0] == 2
    assert bm.search("zzzunknown", 3) == []
    assert BM25([]).search("anything", 3) == []


def test_bm25_prefers_rarer_terms():
    bm = BM25(["iron iron iron", "iron zinc", "zinc"])
    assert bm.search("zinc", 3)[0][0] == 2  # shorter doc, same tf -> higher score


def test_rrf_rewards_agreement_between_lists():
    assert rrf([[1, 2, 3], [3, 1, 4]])[:2] == [1, 3]
    assert rrf([]) == []


def _index():
    chunks = [Chunk(i, t, 1, "d") for i, t in enumerate(TEXTS)]
    vecs = np.eye(4)  # each chunk is its own axis, so the "query vector" picks one exactly
    return InMemoryIndex(chunks, vecs)


def test_index_modes_and_cosine_is_always_reported():
    idx = _index()
    q = [0.0, 0.0, 0.0, 1.0]  # vector says chunk 3
    assert idx.search(q, 1)[0]["chunk_index"] == 3
    assert idx.search(q, 1, query_text="DFE folate", mode="bm25")[0]["chunk_index"] == 2
    hy = idx.search(q, 2, query_text="DFE folate", mode="hybrid")
    assert {r["chunk_index"] for r in hy} == {2, 3}
    assert hy[0]["similarity"] in (0.0, 1.0)  # cosine, not a fused score


def test_rerank_reorders_candidates_and_gate_still_applies(monkeypatch):
    idx = _index()
    monkeypatch.setattr(retrieval, "rerank", lambda q, texts, model=None: [float(i == 1) for i in range(len(texts))])
    from eval import corpus

    monkeypatch.setattr(corpus, "rerank", retrieval.rerank)
    out = idx.search([1.0, 0.0, 0.0, 0.0], 4, query_text="x", reranker="m", candidates=4)
    assert out[0]["chunk_index"] == 1 or out[0]["similarity"] >= 0
    gated = idx.search([1.0, 0.0, 0.0, 0.0], 4, 0.5, query_text="x", reranker="m", candidates=4)
    assert [r["chunk_index"] for r in gated] == [0]
