"""Per-request step timings ("where did the 2.3 seconds go?").

    with trace.span("retrieve"):
        ...
    trace.summary()  -> {"plan": 410, "embed": 180, "retrieve": 35, "generate": 1650}

Same pattern as app/usage.py: a request-scoped ContextVar holding a mutable dict, so spans
recorded deep inside rag.py are attributed to the request without passing anything around.
Repeated steps (e.g. one embed per sub-query) add up.
"""
import time
from contextlib import contextmanager
from contextvars import ContextVar

_current: ContextVar[dict[str, float] | None] = ContextVar("trace", default=None)


def start() -> dict[str, float]:
    t: dict[str, float] = {}
    _current.set(t)
    return t


@contextmanager
def span(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t = _current.get()
        if t is not None:
            t[name] = t.get(name, 0.0) + (time.perf_counter() - t0) * 1000


def summary() -> dict[str, int]:
    t = _current.get() or {}
    return {k: round(v) for k, v in t.items()}
