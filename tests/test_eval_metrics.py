import math

from eval.metrics import (
    auc,
    bootstrap_ci,
    contains_evidence,
    evidence_ranks,
    normalize,
    recall_at_k,
    reciprocal_rank,
)

EV = "Vitamin C helps heal wounds and maintain healthy bones"


def test_normalize_ignores_punctuation_glyphs_and_case():
    assert normalize("Vitamin C: � Helps  heal-wounds!") == "vitaminchelpshealwounds"


def test_contains_evidence_survives_pdf_noise():
    chunk = "Vitamin C � Function: Helps heal wounds, and maintain healthy bones. Food sources..."
    assert contains_evidence(chunk, "Helps heal wounds, and maintain healthy bones")
    assert not contains_evidence(chunk, "Iron prevents anaemia in pregnant women")


def test_too_short_evidence_never_matches():
    assert not contains_evidence("vitamin c is here", "vitamin c")


def test_evidence_ranks_and_recall():
    retrieved = ["unrelated text about proteins and amino acids", EV + " in the body", "other"]
    ranks = evidence_ranks(retrieved, [EV, "A second evidence sentence that is never retrieved"])
    assert ranks == [2, None]
    assert recall_at_k(ranks, 1) == 0.0
    assert recall_at_k(ranks, 2) == 0.5
    assert recall_at_k(ranks, 10) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank([1, None]) == 1.0
    assert reciprocal_rank([None, 4]) == 0.25
    assert reciprocal_rank([None]) == 0.0
    assert reciprocal_rank([12], cutoff=10) == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    vals = [1.0] * 40 + [0.0] * 10
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.8 < hi
    assert (lo, hi) == bootstrap_ci(vals)
    assert math.isnan(bootstrap_ci([])[0])


def test_auc():
    assert auc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5
    assert math.isnan(auc([], [0.5]))
