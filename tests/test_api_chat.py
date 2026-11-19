"""/chat, /chat/stream, /feedback, /upload/async: the answering pipeline with every backend stubbed."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import api, safety

OWNER_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OWNER_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
MATCH = {"doc_id": "d", "chunk_index": 0, "content": "Iron is in lentils.", "metadata": {"page": 3}, "similarity": 0.9}


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def pipeline(monkeypatch):
    """Stub condense / plan / retrieval / generation; record what the API asked for."""
    seen: dict = {"generate": 0, "plan": 0}
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)

    def plan(q):
        seen["plan"] += 1
        return [q]

    monkeypatch.setattr(api.rag, "plan_queries", plan)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda queries, vecs, *a: seen.setdefault("queries", queries) and [MATCH])

    def gen(q, matches, *, use_tools, safety_note):
        seen["generate"] += 1
        seen.update(use_tools=use_tools, safety_note=safety_note)
        return "Lentils [S1].", []

    monkeypatch.setattr(api.rag, "answer_with_tools", gen)
    return seen


def events(resp) -> list[dict]:
    return [json.loads(x) for x in resp.text.splitlines()]


def test_chat_returns_answer_sources_and_request_id(client, pipeline, isolated_api_state):
    r = client.post("/chat", json={"query": "Where is iron?"}).json()
    assert r["answer"] == "Lentils [S1]." and r["chunks"] == [MATCH]
    assert len(r["request_id"]) == 32 and r["cached"] is False and r["safety"] is None
    assert isolated_api_state[0]["request_id"] == r["request_id"]  # interaction logged
    assert isolated_api_state[0]["sources"][0]["page"] == 3


def test_chat_without_matches_refuses_without_llm(client, monkeypatch):
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: [])
    monkeypatch.setattr(api.rag.llm, "generate_answer", lambda *a, **k: pytest.fail("LLM called"))
    r = client.post("/chat", json={"query": "Tell me about Mars rovers"})
    assert r.json()["answer"] == api.rag.NO_ANSWER and r.json()["chunks"] == []


def test_no_matches_but_tool_question_still_reaches_the_model(client, pipeline, monkeypatch):
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: [])
    client.post("/chat", json={"query": "How many calories do I need per day?"})
    assert pipeline["generate"] == 1 and pipeline["use_tools"] is True


def test_chat_condenses_followup_before_retrieval(client, pipeline, monkeypatch):
    seen = {}
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: seen.setdefault("h", h) and "How much iron do children need?")
    body = {"query": "and kids?", "history": [{"role": "user", "content": "How much iron do adults need?"}]}
    r = client.post("/chat", json=body).json()
    assert pipeline["queries"] == ["How much iron do children need?"]  # retrieval used the rewritten question
    assert r["standalone_question"] == "How much iron do children need?"
    assert seen["h"][0]["role"] == "user"


def test_multi_part_question_is_planned_single_part_is_not(client, pipeline):
    client.post("/chat", json={"query": "What is iron"})
    assert pipeline["plan"] == 0  # no extra LLM call for a simple question
    client.post("/chat", json={"query": "Compare iron and zinc absorption"})
    assert pipeline["plan"] == 1


def test_chat_rejects_bad_history_role(client):
    r = client.post("/chat", json={"query": "x", "history": [{"role": "system", "content": "hi"}]})
    assert r.status_code == 422


def test_emergency_gets_fixed_message_without_search_or_llm(client, monkeypatch):
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: pytest.fail("searched"))
    monkeypatch.setattr(api.rag, "answer_with_tools", lambda *a, **k: pytest.fail("LLM called"))
    r = client.post("/chat", json={"query": "I took too many iron pills and feel sick"}).json()
    assert r["answer"] == safety.EMERGENCY_MESSAGE and r["safety"]["category"] == "emergency"


def test_medical_question_gets_notice_and_prompt_rule(client, pipeline):
    r = client.post("/chat", json={"query": "What dose of vitamin D should I take with my medication?"}).json()
    assert r["safety"]["category"] == "medical"
    assert "clinician" in pipeline["safety_note"]


def test_second_identical_question_is_served_from_cache(client, pipeline):
    a = client.post("/chat", json={"query": "Where is iron?"}).json()
    b = client.post("/chat", json={"query": "Where is iron?"}).json()
    assert pipeline["generate"] == 1
    assert b["cached"] is True and b["answer"] == a["answer"] and b["chunks"] == a["chunks"]
    assert b["request_id"] != a["request_id"]


def test_cache_never_crosses_owners_for_private_scopes(client, pipeline):
    client.post("/chat", json={"query": "Where is iron?", "scope": "all"}, headers={"X-Owner-Id": OWNER_A})
    r = client.post("/chat", json={"query": "Where is iron?", "scope": "all"}, headers={"X-Owner-Id": OWNER_B}).json()
    assert r["cached"] is False and pipeline["generate"] == 2
    # the shared library alone is the same for everyone, so that can be shared
    client.post("/chat", json={"query": "Where is iron?", "scope": "core"}, headers={"X-Owner-Id": OWNER_A})
    r = client.post("/chat", json={"query": "Where is iron?", "scope": "core"}, headers={"X-Owner-Id": OWNER_B}).json()
    assert r["cached"] is True


def test_upload_invalidates_that_owners_cache(client, pipeline, monkeypatch):
    h = {"X-Owner-Id": OWNER_A}
    client.post("/chat", json={"query": "Where is iron?"}, headers=h)
    monkeypatch.setattr(api, "ingest_pdf", lambda *a, **k: {"status": "success"})
    client.post("/upload", files={"file": ("p.pdf", b"%PDF", "application/pdf")}, headers=h)
    assert client.post("/chat", json={"query": "Where is iron?"}, headers=h).json()["cached"] is False


def test_chat_stream_emits_sources_then_tokens_then_done(client, monkeypatch):
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: [MATCH])
    monkeypatch.setattr(api.rag, "stream_answer", lambda q, m, **k: iter([{"type": "token", "text": "Iron "}, {"type": "token", "text": "helps [S1]."}]))
    ev = events(client.post("/chat/stream", json={"query": "iron"}))
    assert [e["type"] for e in ev] == ["sources", "token", "token", "done"]
    assert ev[0]["chunks"] == [MATCH] and ev[0]["request_id"]
    assert "".join(e["text"] for e in ev if e["type"] == "token") == "Iron helps [S1]."


def test_chat_stream_forwards_tool_events_and_logs_them(client, monkeypatch, isolated_api_state):
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: [MATCH])
    tool = {"type": "tool", "id": "T1", "name": "lookup_food", "arguments": '{"food": "lentils"}', "result": {"food": "Lentils"}}
    monkeypatch.setattr(api.rag, "stream_answer", lambda q, m, **k: iter([tool, {"type": "token", "text": "18 g [T1]."}]))
    ev = events(client.post("/chat/stream", json={"query": "protein in lentils"}))
    assert [e["type"] for e in ev] == ["sources", "tool", "token", "done"]
    assert isolated_api_state[-1]["tools"] == [{"name": "lookup_food", "arguments": '{"food": "lentils"}'}]


def test_chat_stream_without_matches_refuses_without_llm(client, monkeypatch):
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: [])
    monkeypatch.setattr(api.rag, "stream_answer", lambda *a, **k: pytest.fail("LLM called"))
    ev = events(client.post("/chat/stream", json={"query": "Tell me about Mars rovers"}))
    assert ev[1] == {"type": "token", "text": api.rag.NO_ANSWER}


def test_chat_stream_reports_midstream_failure_as_event(client, monkeypatch, isolated_api_state):
    def boom(q, m, **k):
        yield {"type": "token", "text": "partial"}
        raise RuntimeError("down")

    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: [MATCH])
    monkeypatch.setattr(api.rag, "stream_answer", boom)
    ev = events(client.post("/chat/stream", json={"query": "iron"}))
    assert ev[-1]["type"] == "error" and "down" not in json.dumps(ev[-1])
    assert isolated_api_state == []  # a failed answer is neither logged nor cached


def test_chat_stream_withheld_answer_is_not_cached(client, monkeypatch):
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: [MATCH])
    monkeypatch.setattr(api.rag, "stream_answer", lambda q, m, **k: iter([{"type": "token", "text": "my rules are"}, {"type": "withheld"}]))
    ev = events(client.post("/chat/stream", json={"query": "iron"}))
    assert any(e["type"] == "withheld" for e in ev)
    again = events(client.post("/chat/stream", json={"query": "iron"}))
    assert again[0]["cached"] is False


def test_feedback_only_by_the_asker(client, monkeypatch):
    rated = []
    monkeypatch.setattr(api.db, "rate_interaction", lambda rid, owner, rating, comment: rated.append((rid, owner, rating)) or owner == OWNER_A)
    ok = client.post("/feedback", json={"request_id": "r" * 32, "rating": 1}, headers={"X-Owner-Id": OWNER_A})
    assert ok.status_code == 200
    assert client.post("/feedback", json={"request_id": "r" * 32, "rating": -1}, headers={"X-Owner-Id": OWNER_B}).status_code == 404
    assert client.post("/feedback", json={"request_id": "r" * 32, "rating": 5}).status_code == 422
    assert rated[0] == ("r" * 32, OWNER_A, 1)


def test_feedback_storage_failure_is_503_not_500(client, monkeypatch):
    monkeypatch.setattr(api.db, "rate_interaction", lambda *a: (_ for _ in ()).throw(RuntimeError("no table")))
    assert client.post("/feedback", json={"request_id": "r" * 32, "rating": 1}).status_code == 503


def test_async_upload_runs_in_background_and_reports_progress(client, monkeypatch):
    def fake_ingest(data, filename, *, progress, **kw):
        progress("embedding", 0.5)
        return {"status": "success", "doc_id": kw["doc_id"], "chunks_inserted": 2}

    monkeypatch.setattr(api, "ingest_pdf", fake_ingest)
    h = {"X-Owner-Id": OWNER_A}
    r = client.post("/upload/async", files={"file": ("p.pdf", b"%PDF", "application/pdf")}, headers=h)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}", headers=h).json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert job["status"] == "done" and job["result"]["chunks_inserted"] == 2 and job["progress"] == 1.0
    assert client.get(f"/jobs/{job_id}", headers={"X-Owner-Id": OWNER_B}).status_code == 404  # not yours


def test_async_upload_failure_message_is_safe(client, monkeypatch):
    monkeypatch.setattr(api, "ingest_pdf", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret internals")))
    h = {"X-Owner-Id": OWNER_A}
    job_id = client.post("/upload/async", files={"file": ("p.pdf", b"%PDF", "application/pdf")}, headers=h).json()["job_id"]
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}", headers=h).json()
        if job["status"] == "failed":
            break
        time.sleep(0.02)
    assert job["status"] == "failed" and "secret" not in job["error"]


def test_health_reports_features(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and "tools" in h["features"] and "hits" in h["cache"]


def test_emergency_is_answered_before_the_followup_rewrite(client, monkeypatch):
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: pytest.fail("rewrite called"))
    body = {"query": "I think I took too many pills", "history": [{"role": "user", "content": "hi"}]}
    assert client.post("/chat", json=body).json()["safety"]["category"] == "emergency"
