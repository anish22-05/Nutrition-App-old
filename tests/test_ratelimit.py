from fastapi.testclient import TestClient

from app import api
from app.ratelimit import RateLimiter


def test_sliding_window_allows_then_blocks_then_recovers():
    now = [0.0]
    rl = RateLimiter(clock=lambda: now[0])
    assert [rl.allow("k", 2, 60) for _ in range(3)] == [True, True, False]
    assert rl.allow("other", 2, 60)  # keys are independent
    now[0] = 61.0
    assert rl.allow("k", 2, 60)  # window slid


def test_chat_endpoint_returns_429_when_over_limit(monkeypatch):
    monkeypatch.setattr(api, "limiter", RateLimiter())
    monkeypatch.setitem(api.LIMITED, "/chat", 2)
    monkeypatch.setattr(api.rag, "retrieve_many", lambda *a: [])
    monkeypatch.setattr(api.rag, "standalone_question", lambda h, q: q)
    c = TestClient(api.app)
    codes = [c.post("/chat", json={"query": "iron"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert c.get("/health").status_code == 200  # unlimited endpoints unaffected
