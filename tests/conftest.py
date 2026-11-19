import hashlib

import numpy as np
import pytest

from app import api


def fake_vector(text: str) -> list[float]:
    """Deterministic, effectively unique per text (random 64-d directions are nearly orthogonal)."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
    return np.random.default_rng(seed).normal(size=64).tolist()
from app.cache import SemanticCache
from app.ratelimit import RateLimiter


@pytest.fixture(autouse=True)
def fresh_rate_limiter(monkeypatch):
    """Every test starts with an empty limiter so request counts never leak between tests."""
    monkeypatch.setattr(api, "limiter", RateLimiter())


@pytest.fixture(autouse=True)
def isolated_api_state(request, monkeypatch):
    """API tests never touch the network: embeddings are fake, interaction logging goes to a
    list, and each test gets an empty answer cache. (Other test modules patch what they need.)"""
    logged: list[dict] = []
    monkeypatch.setattr(api, "cache", SemanticCache())
    monkeypatch.setattr(api.db, "log_interaction", logged.append)
    if request.module.__name__.split(".")[-1].startswith(("test_api", "test_ratelimit")):
        monkeypatch.setattr(api.llm, "embed_texts", lambda texts: [fake_vector(t) for t in texts])
    return logged
