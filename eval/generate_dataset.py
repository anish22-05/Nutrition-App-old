"""Draft a golden Q&A set from the corpus.   python -m eval.generate_dataset

Three kinds of questions, because each catches a different failure:
  single        - answer lives in one chunk (basic retrieval quality)
  multi         - needs two related chunks from different pages (recall of *all* evidence)
  unanswerable  - not in the document (does the system admit it instead of hallucinating?)

An LLM drafts the answerable ones; code then rejects anything that fails mechanical checks
(evidence must be a verbatim quote, question must stand alone). A human still reads the
result, because LLM-written tests have blind spots.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.ingestion import count_tokens
from eval.corpus import build_index
from eval.llm_json import chat_json
from eval.metrics import MIN_EVIDENCE_CHARS, contains_evidence, normalize

DATA = Path(__file__).parent / "data"
DEFAULT_PDF = Path("data/introduction-to-nutrition.pdf")

# Boilerplate printed on every slide of the deck. Stripped before the question writer sees
# the passage so it does not write questions about the footer.
BOILERPLATE = re.compile(
    r"EU-India Capacity Building Initiative for Trade Development \(CITD\)\s*\S*\s*Lot 1 SPS/TBT/Customs"
)

# A question that points at "the passage/slide/document" is useless: real users never say that.
LEAKY = re.compile(
    r"\b(passage|slide|document|text|excerpt|context|section|above|following|according to|mentioned)\b", re.IGNORECASE
)

SINGLE_SYSTEM = """You write evaluation questions for a nutrition Q&A system.
Given a PASSAGE, write ONE question that a nutrition student could ask and that the passage answers.

Rules:
- The question must be fully self-contained. Never refer to "the passage", "the text", "the slide", "the document" or "this section".
- Paraphrase: do not copy long phrases from the passage into the question (that would make retrieval trivially easy).
- Ask about a concrete fact, definition, list or relationship - not an opinion.
- "answer": 1-2 sentences, using only facts in the passage.
- "evidence": copy, EXACTLY and verbatim, the 1-2 consecutive sentences from the passage that contain the answer.
- If the passage has no substantive, askable fact (title slide, filler, fragments), return {"skip": true}.
Return JSON: {"skip": bool, "question": str, "answer": str, "evidence": str}"""

MULTI_SYSTEM = """You write evaluation questions for a nutrition Q&A system.
You get two PASSAGES (A and B) from different parts of a document. Write ONE question that can only be
answered fully by combining a fact from A with a fact from B (e.g. compare, relate, or list across both).

Rules:
- Self-contained: never mention "passage", "text", "slide", "document", "A" or "B".
- Paraphrase instead of copying phrases.
- "answer": 2-3 sentences using only facts from the passages.
- "evidence_a" / "evidence_b": EXACT verbatim 1-2 sentences copied from A and from B respectively.
- If the two passages cannot be meaningfully combined into one natural question, return {"skip": true}.
Return JSON: {"skip": bool, "question": str, "answer": str, "evidence_a": str, "evidence_b": str}"""

# (question, regex that must NOT appear in the corpus - otherwise it might actually be answerable)
UNANSWERABLE = [
    ("What is the recommended weekly dose of semaglutide for weight loss?", r"semaglutide|ozempic"),
    ("How does a ketogenic diet induce ketosis and what are its health risks?", r"keto"),
    ("Which probiotic strains work best for lactose intolerance?", r"probiotic"),
    ("What are the typical symptoms of celiac disease?", r"celiac|coeliac"),
    ("How much caffeine is in a cup of espresso?", r"caffeine|espresso"),
    ("What is the carbon footprint of beef production per kilogram?", r"carbon|footprint"),
    ("How many hours should someone fast on a 16:8 intermittent fasting plan?", r"fasting"),
    ("How much omega-3 does a tablespoon of flaxseed oil contain?", r"flax"),
    ("What protein intake do professional bodybuilders need per kilogram of bodyweight?", r"bodybuild"),
    ("Who won the FIFA World Cup in 2018?", r"world cup|fifa"),
    ("How do I reset the password on my home Wi-Fi router?", r"router|wi-?fi"),
    ("What is the capital city of Australia?", r"canberra|australia"),
]


def _clean(text: str) -> str:
    return BOILERPLATE.sub(" ", text).replace("\ufffd", "-").strip()


def _valid_question(q: str) -> bool:
    return 15 <= len(q) <= 300 and q.endswith("?") and not LEAKY.search(q)


def draft_single(index, n: int, rng: random.Random, model: str) -> list[dict]:
    pool = [c for c in index.chunks if count_tokens(_clean(c.text)) >= 40]
    rng.shuffle(pool)
    out: list[dict] = []
    for chunk in pool:
        if len(out) >= n:
            break
        res = chat_json(SINGLE_SYSTEM, f"PASSAGE:\n{_clean(chunk.text)}", model)
        if res.get("skip"):
            continue
        q, ev = (res.get("question") or "").strip(), (res.get("evidence") or "").strip()
        if _valid_question(q) and len(normalize(ev)) >= MIN_EVIDENCE_CHARS and contains_evidence(chunk.text, ev):
            out.append(
                {
                    "type": "single",
                    "question": q,
                    "reference_answer": (res.get("answer") or "").strip(),
                    "evidence": [ev],
                    "pages": [chunk.page],
                }
            )
    return out


def draft_multi(index, n: int, rng: random.Random, model: str) -> list[dict]:
    """Pair each sampled chunk with its nearest neighbour on a *different, non-adjacent* page."""
    sims = index.vectors @ index.vectors.T
    order = list(range(len(index.chunks)))
    rng.shuffle(order)
    out: list[dict] = []
    used: set[int] = set()
    for i in order:
        if len(out) >= n:
            break
        a = index.chunks[i]
        if i in used or count_tokens(_clean(a.text)) < 40:
            continue
        cands = [int(j) for j in np.argsort(-sims[i]) if abs(index.chunks[j].page - a.page) > 2 and int(j) not in used]
        if not cands:
            continue
        b = index.chunks[cands[0]]
        res = chat_json(MULTI_SYSTEM, f"PASSAGE A:\n{_clean(a.text)}\n\nPASSAGE B:\n{_clean(b.text)}", model)
        if res.get("skip"):
            continue
        q = (res.get("question") or "").strip()
        ea, eb = (res.get("evidence_a") or "").strip(), (res.get("evidence_b") or "").strip()
        if _valid_question(q) and contains_evidence(a.text, ea) and contains_evidence(b.text, eb):
            used |= {i, cands[0]}
            out.append(
                {
                    "type": "multi",
                    "question": q,
                    "reference_answer": (res.get("answer") or "").strip(),
                    "evidence": [ea, eb],
                    "pages": [a.page, b.page],
                }
            )
    return out


def draft_unanswerable(index) -> list[dict]:
    corpus = " ".join(c.text.lower() for c in index.chunks)
    out = []
    for q, forbidden in UNANSWERABLE:
        if re.search(forbidden, corpus):
            print(f"  dropping unanswerable candidate (keyword found in corpus): {q}")
            continue
        out.append({"type": "unanswerable", "question": q, "reference_answer": None, "evidence": [], "pages": []})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--single", type=int, default=45)
    ap.add_argument("--multi", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=DATA / "golden_draft.jsonl")
    args = ap.parse_args()

    model = get_settings().judge_model
    rng = random.Random(args.seed)
    index = build_index(args.pdf)
    print(f"corpus: {len(index.chunks)} chunks from {args.pdf.name}")

    items = draft_single(index, args.single, rng, model) + draft_multi(index, args.multi, rng, model)
    items += draft_unanswerable(index)

    seen: set[str] = set()
    final = []
    for it in items:
        key = normalize(it["question"])
        if key in seen:
            continue
        seen.add(key)
        final.append({"id": f"q{len(final) + 1:03d}", **it})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in final) + "\n", encoding="utf-8")
    counts = {t: sum(x["type"] == t for x in final) for t in ("single", "multi", "unanswerable")}
    print(f"wrote {len(final)} drafts to {args.out}  {counts}")


if __name__ == "__main__":
    main()
