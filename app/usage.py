"""Per-request token/cost accounting.

A request-scoped counter (a ContextVar holding a mutable object) collects what each OpenAI call
used; the API middleware logs the total. FastAPI copies the context into its worker threads, and
the copy still points at the same object, so calls made deep in llm.py are attributed correctly.
Prices are USD per 1M tokens and are estimates: update them when OpenAI changes its price list.
"""
from contextvars import ContextVar
from dataclasses import dataclass

PRICES = {  # model -> (input, output) USD per 1M tokens
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        p_in, p_out = PRICES.get(model, (0.0, 0.0))  # unknown model: counted as free, never crashes
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += (input_tokens * p_in + output_tokens * p_out) / 1_000_000
        self.calls += 1


_current: ContextVar[Usage | None] = ContextVar("usage", default=None)


def start() -> Usage:
    u = Usage()
    _current.set(u)
    return u


def record(model: str, input_tokens: int | None, output_tokens: int | None = 0) -> None:
    u = _current.get()
    if u is not None:  # outside a request (CLI, eval) there is nothing to attribute to
        u.add(model, input_tokens or 0, output_tokens or 0)


def summary() -> str:
    u = _current.get()
    if u is None or not u.calls:
        return ""
    return f"{u.calls} OpenAI calls, {u.input_tokens}+{u.output_tokens} tokens, ~${u.cost_usd:.5f}"


def current_cost() -> float | None:
    u = _current.get()
    return round(u.cost_usd, 6) if u is not None else None
