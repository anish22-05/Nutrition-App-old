from app import rag


def test_answer_refuses_without_calling_llm_when_no_context(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called when nothing was retrieved")

    monkeypatch.setattr(rag.llm, "generate_answer", boom)
    assert rag.answer("anything?", []) == rag.NO_ANSWER


def test_answer_delegates_to_llm_with_context(monkeypatch):
    monkeypatch.setattr(rag.llm, "generate_answer", lambda q, m, **kw: (f"{q}|{len(m)}|{kw['use_tools']}", []))
    assert rag.answer("q?", [{"content": "x"}]) == "q?|1|False"  # tools stay off unless asked (the eval relies on it)


def test_retrieve_reranks_candidates_when_enabled(monkeypatch):
    from app import rag

    cands = [{"content": c, "similarity": 0.9 - i / 10} for i, c in enumerate(["a", "b", "c"])]
    seen = {}
    monkeypatch.setattr(rag.llm, "embed_texts", lambda t: [[0.0]])
    monkeypatch.setattr(rag.db, "search_chunks", lambda vec, k, *a: seen.setdefault("k", k) and cands)
    monkeypatch.setattr(rag, "rerank", lambda q, texts, m: [0.1, 0.9, 0.5])
    monkeypatch.setattr(rag.get_settings(), "rerank_enabled", True)
    out = rag.retrieve("q", 2)
    assert [c["content"] for c in out] == ["b", "c"]  # re-ordered, truncated to top_k
    assert seen["k"] >= 30  # asked the database for a wider candidate list
