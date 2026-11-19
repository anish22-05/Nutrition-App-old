"""Retrieval metrics. Pure functions - no network, no DB, fully unit-tested.

Relevance is decided by *evidence text*, not by chunk id: a retrieved chunk is relevant
if it contains the gold evidence sentence(s). That makes the golden set survive any change
to chunking (sizes, overlap, strategy) - which is exactly what Phase 2 experiments change.
"""
import re

import numpy as np

MIN_EVIDENCE_CHARS = 25  # shorter than this and a substring match is meaningless
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and drop everything but letters/digits, so matching ignores the bullet
    glyphs, hyphenation, quotes and whitespace that PDF extraction mangles."""
    return _NON_ALNUM.sub("", text.lower())


def contains_evidence(chunk_text: str, evidence: str) -> bool:
    ev = normalize(evidence)
    return len(ev) >= MIN_EVIDENCE_CHARS and ev in normalize(chunk_text)


def evidence_ranks(retrieved_texts: list[str], evidence: list[str]) -> list[int | None]:
    """For each gold evidence item: 1-based rank of the first retrieved chunk containing it."""
    norm = [normalize(t) for t in retrieved_texts]
    ranks: list[int | None] = []
    for ev in evidence:
        e = normalize(ev)
        ranks.append(next((i + 1 for i, t in enumerate(norm) if e in t), None))
    return ranks


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of gold evidence items found in the top-k. For single-evidence questions
    this is just 'did we retrieve it' (a.k.a. hit rate)."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def reciprocal_rank(ranks: list[int | None], cutoff: int = 10) -> float:
    """1/rank of the first relevant chunk (0 if none within the cutoff)."""
    found = [r for r in ranks if r is not None and r <= cutoff]
    return 1.0 / min(found) if found else 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval for the mean. With ~50 questions a single number
    like 0.84 is noisy; the interval is what tells you whether a change is real."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def auc(positive: list[float], negative: list[float]) -> float:
    """P(random positive scores higher than random negative); ties count half.
    Used to ask: can the top-1 similarity score tell answerable from unanswerable questions?"""
    if not positive or not negative:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")
