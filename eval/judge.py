"""LLM-as-judge for the two things you cannot check with string matching:
is the answer *grounded* in the retrieved context, and is it *correct* versus the reference?

Known weaknesses (worth stating in any write-up): a judge model has its own biases, tends to
favour answers that look like its own, and is not perfectly repeatable. We reduce that with
temperature 0, narrow yes/no style questions instead of vague 1-10 scores, and by keeping the
judge model configurable (`JUDGE_MODEL`) so it can be a stronger model than the one being judged.
"""
import re

from app.llm import format_context
from app.rag import NO_ANSWER
from eval.llm_json import chat_json

FAITHFULNESS_SYSTEM = """You check whether an ANSWER is supported by the CONTEXT it was written from.
1. Split the ANSWER into atomic factual claims. Ignore citation tags like [S1] and ignore any refusal sentence.
2. For each claim decide if the CONTEXT directly supports it. Use only the CONTEXT, never your own knowledge.
Return JSON: {"claims": [{"claim": str, "supported": bool}]}"""

CORRECTNESS_SYSTEM = """You compare an ANSWER with a REFERENCE answer for a QUESTION.
1. Split the REFERENCE into its key facts (usually 1-4).
2. For each key fact say whether the ANSWER states it ("present") and whether the ANSWER says something that
   contradicts it ("contradicted"). Judge ONLY the REFERENCE's facts. Extra information in the ANSWER that the
   REFERENCE does not mention is irrelevant here: never mention it and never count it against the ANSWER.
Return JSON: {"facts": [{"fact": str, "present": bool, "contradicted": bool}]}"""

CITATION = re.compile(r"\[S(\d+)\]")


def is_abstention(answer: str) -> bool:
    """A clean refusal is the refusal sentence and nothing else (that is what the prompt asks for).
    Substring matching would also catch a correct partial answer that merely ends with the
    sentence, and would let 'refusal + hallucination' count as a successful abstention."""
    return answer.strip().rstrip(".").lower() == NO_ANSWER.rstrip(".").lower()


def valid_citations(answer: str, n_sources: int) -> bool:
    """True if the answer cites at least one source and every cited [Sn] actually exists."""
    cited = [int(n) for n in CITATION.findall(answer)]
    return bool(cited) and all(1 <= n <= n_sources for n in cited)


def faithfulness(answer: str, matches: list[dict], model: str) -> float | None:
    """Share of the answer's claims supported by the context (None if it makes no claims)."""
    res = chat_json(
        FAITHFULNESS_SYSTEM, f"CONTEXT:\n{format_context(matches)}\n\nANSWER:\n{answer}", model
    )
    claims = res.get("claims") or []
    if not claims:
        return None
    return sum(bool(c.get("supported")) for c in claims) / len(claims)


def score_facts(facts: list[dict]) -> tuple[float, str]:
    """Verdict computed in code from the judge's per-fact checks, so the judge cannot let
    'the answer says more than the reference' leak into the grade."""
    if not facts:
        return 0.0, "judge returned no facts"
    if any(f.get("contradicted") for f in facts):
        return 0.0, "contradicts: " + "; ".join(str(f.get("fact")) for f in facts if f.get("contradicted"))
    present = [bool(f.get("present")) for f in facts]
    if all(present):
        return 1.0, ""
    missing = [str(f.get("fact")) for f in facts if not f.get("present")]
    return (0.5 if any(present) else 0.0), "missing: " + "; ".join(missing)


def correctness(question: str, answer: str, reference: str, model: str) -> tuple[float, str]:
    prompt = f"QUESTION: {question}\n\nREFERENCE: {reference}\n\nANSWER: {answer}"
    res = chat_json(CORRECTNESS_SYSTEM, prompt, model)
    return score_facts(res.get("facts") or [])
