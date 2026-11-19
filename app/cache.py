"""Semantic answer cache: reuse an answer when a new question means the same thing.

The question is embedded for retrieval anyway, so checking it against cached question vectors is
almost free, while a hit skips retrieval *and* the LLM call (the slow, costly part). Popular
questions ("how much protein do I need?", "how much protein should I eat?") hit it a lot.

Correctness rules:
* An answer is only reused within the same *visibility key*: scope, document filter, top_k,
  tools on/off, safety category, and - for anything that can include private documents - the
  owner. One user's private answer is never served to another.
* A user's entries are dropped when they upload or delete a document (their sources changed).
  Core-library changes come from the admin CLI in another process, so entries also expire (TTL).
* Multi-turn context matters only through the standalone (rewritten) question, which is what is
  embedded here, so follow-ups are safe to cache too.
In-process and bounded (LRU); per container, like the rate limiter.
"""
import threading
import time
from collections import OrderedDict

import numpy as np


class SemanticCache:
    def __init__(self, threshold: float = 0.97, ttl_s: float = 3600, max_entries: int = 500, clock=time.monotonic):
        self.threshold, self.ttl_s, self.max_entries = threshold, ttl_s, max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, tuple[tuple, np.ndarray, float, dict, str | None]] = OrderedDict()
        self._next = 0
        self.hits = self.misses = 0

    @staticmethod
    def _unit(vec) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def get(self, key: tuple, vec) -> dict | None:
        q = self._unit(vec)
        now = self._clock()
        with self._lock:
            best, best_sim = None, self.threshold
            for eid, (k, v, t, payload, _owner) in list(self._entries.items()):
                if now - t > self.ttl_s:
                    del self._entries[eid]
                    continue
                if k == key:
                    sim = float(v @ q)
                    if sim >= best_sim:
                        best, best_sim = eid, sim
            if best is None:
                self.misses += 1
                return None
            self._entries.move_to_end(best)
            self.hits += 1
            return {**self._entries[best][3], "cache_similarity": round(best_sim, 4)}

    def put(self, key: tuple, vec, payload: dict, owner_id: str | None = None) -> None:
        with self._lock:
            self._entries[self._next] = (key, self._unit(vec), self._clock(), payload, owner_id)
            self._next += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def invalidate_owner(self, owner_id: str) -> int:
        with self._lock:
            gone = [eid for eid, e in self._entries.items() if e[4] == owner_id]
            for eid in gone:
                del self._entries[eid]
            return len(gone)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
