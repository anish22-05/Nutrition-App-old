from app.rag import NO_ANSWER
from eval.judge import is_abstention, valid_citations


def test_clean_refusal_is_abstention():
    assert is_abstention(NO_ANSWER)
    assert is_abstention("  " + NO_ANSWER + "\n")


def test_partial_answer_ending_with_refusal_is_not_abstention():
    assert not is_abstention("There are 9 essential amino acids [S1]. " + NO_ANSWER)


def test_refusal_prefixed_hallucination_is_not_abstention():
    assert not is_abstention(NO_ANSWER + " However, Canberra is the capital.")


def test_valid_citations():
    assert valid_citations("Iron prevents anaemia [S1][S3].", 3)
    assert not valid_citations("Iron prevents anaemia [S4].", 3)  # source does not exist
    assert not valid_citations("Iron prevents anaemia.", 3)  # no citation at all


def test_score_facts_ignores_extra_content_and_flags_missing_or_contradicted():
    from eval.judge import score_facts

    assert score_facts([{"fact": "a", "present": True, "contradicted": False}])[0] == 1.0
    assert score_facts([{"fact": "a", "present": True}, {"fact": "b", "present": False}])[0] == 0.5
    assert score_facts([{"fact": "a", "present": False}])[0] == 0.0
    assert score_facts([{"fact": "a", "present": True, "contradicted": True}])[0] == 0.0
    assert score_facts([])[0] == 0.0
