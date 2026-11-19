"""Semantic cache, background jobs, local storage backend, section titles, merging, tracing."""
import time

import numpy as np
import pymupdf
import pytest

from app import local_store, trace
from app.cache import SemanticCache
from app.ingestion import build_chunks_from_pdf, embedding_text, page_sections
from app.jobs import JobStore
from app.retrieval import merge_ranked

# --- cache ---------------------------------------------------------------------------------------


def test_cache_hits_similar_question_with_same_key_only():
    c = SemanticCache(threshold=0.97)
    c.put(("core",), [1.0, 0.0], {"answer": "A"})
    assert c.get(("core",), [1.0, 0.05])["answer"] == "A"   # cos ~ 0.9988
    assert c.get(("core",), [1.0, 0.5]) is None              # cos ~ 0.89: a different question
    assert c.get(("all", "owner-b"), [1.0, 0.0]) is None     # different visibility key
    assert (c.hits, c.misses) == (1, 2)


def test_cache_expiry_owner_invalidation_and_bound():
    now = [0.0]
    c = SemanticCache(ttl_s=10, max_entries=2, clock=lambda: now[0])
    c.put(("k",), [1, 0], {"a": 1}, owner_id="u1")
    now[0] = 11
    assert c.get(("k",), [1, 0]) is None                      # expired
    c.put(("k",), [1, 0], {"a": 2}, owner_id="u1")
    c.put(("j",), [0, 1], {"a": 3})
    c.put(("i",), [1, 1], {"a": 4})                           # evicts the oldest
    assert c.get(("k",), [1, 0]) is None
    assert c.invalidate_owner("u1") == 0 and c.get(("j",), [0, 1])["a"] == 3


# --- jobs ----------------------------------------------------------------------------------------


def wait_for(store, job_id, owner):
    for _ in range(200):
        j = store.get(job_id, owner)
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_job_lifecycle_and_owner_isolation():
    s = JobStore(workers=1)
    seen, ids = [], []
    started = __import__("threading").Event()

    def work(progress):
        started.wait(2)
        progress("embedding", 0.5)
        seen.append(s.get(ids[0], "me")["stage"])
        return {"ok": True}

    jid = s.submit("me", work, str)
    ids.append(jid)
    started.set()
    j = wait_for(s, jid, "me")
    assert j["status"] == "done" and j["result"] == {"ok": True} and seen == ["embedding"]
    assert s.get(jid, "someone-else") is None and "owner_id" not in j


def test_job_failure_uses_error_mapper():
    s = JobStore(workers=1)
    jid = s.submit(None, lambda p: 1 / 0, lambda e: "friendly message")
    assert wait_for(s, jid, None) | {} and s.get(jid, None)["error"] == "friendly message"


# --- local store -----------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store.get_settings(), "local_store_path", str(tmp_path / "s.pkl"))
    monkeypatch.setattr(local_store, "_state", None)
    monkeypatch.setattr(local_store, "_mtime", None)
    return local_store


def rows(*vecs):
    return [{"chunk_index": i, "content": f"c{i}", "metadata": {"page": i}, "embedding": v} for i, v in enumerate(vecs)]


def test_local_store_scopes_and_isolation(store):
    store.replace_document("core-1", "book.pdf", rows([1, 0, 0], [0, 1, 0]), scope="core")
    store.replace_document("u-a", "mine.pdf", rows([1, 0.1, 0]), scope="user", owner_id="A")
    q = [1, 0, 0]
    assert {h["doc_id"] for h in store.search_chunks(q, 5)} == {"core-1"}                  # no owner: core only
    assert {h["doc_id"] for h in store.search_chunks(q, 5, owner_id="A")} == {"core-1", "u-a"}
    assert {h["doc_id"] for h in store.search_chunks(q, 5, owner_id="B")} == {"core-1"}     # not B's
    assert {h["doc_id"] for h in store.search_chunks(q, 5, owner_id="A", scope="mine")} == {"u-a"}
    assert all(h["similarity"] >= 0.9 for h in store.search_chunks(q, 5, min_similarity=0.9, owner_id="A"))
    assert store.list_documents("A")["mine"][0]["doc_id"] == "u-a" and store.count_user_documents("A") == 1


def test_local_store_replace_delete_hash_and_reload(store):
    store.replace_document("d", "a.pdf", rows([1, 0]), scope="core", file_hash="h1")
    store.replace_document("d", "a.pdf", rows([0, 1], [1, 1]), scope="core", file_hash="h2")  # new version
    assert store.get_document("d")["num_chunks"] == 2 and store.find_document_by_hash("h2", "core", None)
    store._state = None  # another process: must reload from disk
    assert len(store.search_chunks([0, 1], 5)) == 2
    store.delete_document("d")
    assert store.search_chunks([0, 1], 5) == [] and store.get_document("d") is None


def test_local_store_interactions_and_rating_owner_check(store):
    store.log_interaction({"request_id": "r1", "owner_id": "A", "question": "q", "answer": "a"})
    assert not store.rate_interaction("r1", "B", 1, None)
    assert store.rate_interaction("r1", "A", -1, "wrong")
    assert store.list_interactions()[0]["comment"] == "wrong"


# --- ingestion: section titles -----------------------------------------------------------------------


def make_pdf_with_outline() -> bytes:
    doc = pymupdf.open()
    for text in ["Iron is a mineral found in lentils and spinach. " * 8, "Zinc supports immune function in adults. " * 8]:
        doc.new_page().insert_text((50, 72), text, fontsize=9)
    note = [[2, "Publisher note", p] for p in (1, 1, 2, 2)]  # boilerplate repeated in the outline
    doc.set_toc([[1, "Minerals", 1], [2, "Iron", 1], note[0], [2, "Zinc", 2], *note[1:]])
    data = doc.tobytes()
    doc.close()
    return data


def test_page_sections_follow_the_outline_and_skip_repeated_titles():
    secs = page_sections(make_pdf_with_outline())
    assert secs == {1: "Minerals > Iron", 2: "Minerals > Zinc"}


def test_chunks_carry_section_and_embedding_text_uses_it():
    _, metas = build_chunks_from_pdf(make_pdf_with_outline(), "m.pdf", min_tokens=5)
    assert {m.get("section") for m in metas} >= {"Minerals > Iron"}
    assert embedding_text("x", {"section": "Minerals > Iron"}) == "Section: Minerals > Iron\nx"
    assert embedding_text("x", {"section": "S"}, contextual=False) == "x"
    assert embedding_text("x", {}) == "x"


# --- merging and tracing -----------------------------------------------------------------------------


def test_merge_ranked_interleaves_and_dedupes():
    a = [{"doc_id": "d", "chunk_index": i} for i in (1, 2, 3)]
    b = [{"doc_id": "d", "chunk_index": i} for i in (9, 1, 8)]
    assert [r["chunk_index"] for r in merge_ranked([a, b], 4)] == [1, 9, 2, 3]


def test_trace_spans_accumulate():
    trace.start()
    with trace.span("embed"):
        np.zeros(3)
    with trace.span("embed"):
        pass
    assert set(trace.summary()) == {"embed"}
