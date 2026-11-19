import numpy as np
import pytest

from eval.corpus import Chunk, InMemoryIndex


def make_index():
    chunks = [Chunk(i, f"chunk {i}", page=i + 1, source="doc.pdf") for i in range(3)]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return InMemoryIndex(chunks, vectors)


def test_search_orders_by_cosine_similarity():
    hits = make_index().search([0.9, 0.1], k=3)
    assert [h["chunk_index"] for h in hits] == [0, 2, 1]
    assert hits[0]["similarity"] > hits[1]["similarity"] > hits[2]["similarity"]


def test_search_respects_k_and_min_similarity():
    idx = make_index()
    assert len(idx.search([1, 0], k=1)) == 1
    assert [h["chunk_index"] for h in idx.search([1, 0], k=3, min_similarity=0.5)] == [0, 2]


def test_result_shape_matches_match_documents_rpc():
    hit = make_index().search([1, 0], k=1)[0]
    assert {"doc_id", "chunk_index", "content", "metadata", "similarity"} <= hit.keys()
    assert hit["metadata"]["page"] == 1


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        InMemoryIndex([Chunk(0, "a", 1, "d")], np.zeros((2, 2)))
