"""Tiny in-process sliding-window rate limiter.

Why this exists: /upload and /chat spend real money (embeddings, LLM tokens) and the identity
is just a random UUID anyone can mint, so the limit is keyed on the *client IP* as well as the
UUID. In-memory means it resets on restart and is per-process; behind several workers you would
move it to Redis or the gateway. That trade-off is fine for a single-container deploy.
"""
import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._clock = clock

    def allow(self, key: str, limit: int, window_s: float) -> bool:
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory: drop idle keys
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= window_s]:
                    del self._hits[k]
            return True
