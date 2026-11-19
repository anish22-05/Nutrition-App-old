"""Turn golden_textbook_draft.jsonl into golden.jsonl.   python -m eval.curate_textbook

Human-in-the-loop step, written down so it is reviewable: every draft question was read, and
those dropped are listed with a reason. Unanswerable questions are added here (the textbook is
broad, so many obvious "not in the document" candidates turned out to be in it - each one is
checked against the corpus text before being kept).
"""
import json
import re

from eval.corpus import load_chunks
from eval.generate_dataset import DATA
from eval.metrics import contains_evidence

PDF = "data/Human-Nutrition-2020.pdf"

DROP = {
    "q030": "vague: 'research study methods' has many valid chapters",
    "q050": "vague advice question; any meat page could answer it",
    "q055": "vague: many pages recommend protein sources",
    "q085": "opinion/advice question with no single correct answer",
    "q083": "duplicate of q063 (niacin deficiency symptoms are described on two pages)",
    "q059": "duplicate of q058 (same breast-milk passage)",
    "q107": "duplicate of q108/q044 (Adequate Intake is defined on several pages)",
    "q108": "same as q107",
    "q102": "overlaps q045/q074 (osteoporosis is covered in several places)",
    "q074": "same as q102",
    "q112": "multi-chunk pairing contrived; two unrelated passages",
    "q116": "multi-chunk pairing contrived",
    "q117": "multi-chunk question answered by one passage",
    "q120": "multi-chunk pairing contrived",
    "q122": "multi-chunk pairing contrived",
    "q126": "multi-chunk question too vague",
    "q127": "multi-chunk pairing contrived (lysosomes vs digestive system)",
    "q128": "multi-chunk pairing contrived",
    "q129": "overlaps q106 (daily water output)",
    "q130": "multi-chunk pairing contrived",
    "q131": "multi-chunk pairing contrived",
    "q135": "multi-chunk pairing contrived (nursing diet vs allergies)",
}

# (question, regex that must NOT appear in the corpus). Mix of clearly off-topic and "near-miss":
# nutrition-flavoured questions the textbook does not actually answer, which is the hard case.
UNANSWERABLE = [
    ("What is the recommended weekly dose of semaglutide for weight loss?", r"semaglutide|ozempic"),
    ("Who won the FIFA World Cup in 2018?", r"world cup|fifa"),
    ("How do I reset the password on my home Wi-Fi router?", r"router|wi-?fi"),
    ("What is the standard starting dose of metformin for type 2 diabetes?", r"metformin"),
    ("How many calories are in a Starbucks venti caramel frappuccino?", r"starbucks|frappuccino"),
    ("What is the recommended daily dose of ashwagandha for stress?", r"ashwagandha"),
    ("Which CRISPR approach is used to treat sickle cell disease?", r"crispr"),
    ("How do I install Python packages with pip?", r"\bpip\b|python"),
    ("What is the GDP of Brazil?", r"\bgdp\b|brazil"),
    ("What does the Whole30 programme allow for breakfast?", r"whole30"),
    ("How much creatine monohydrate should an athlete take per day?", r"creatine"),
    ("What are the side effects of the drug orlistat?", r"orlistat"),
    ("How is a Big Mac's nutritional content split between fat and protein?", r"big mac|mcdonald"),
    ("What is the glycaemic index of Ezekiel bread?", r"ezekiel"),
    ("Which gut microbiome tests are most accurate for personalised diets?", r"microbiome test|stool test"),
    ("What is the recommended dose of berberine for blood sugar?", r"berberine"),
    ("Explain how the Mediterranean diet affects Alzheimer's disease progression in clinical trials.", r"alzheimer"),
    ("How does the DASH diet compare with the TLC diet in randomised trials for LDL reduction?", r"\bTLC\b"),
    ("What is the best way to change a flat car tyre?", r"tyre|tire"),
    ("What is the half-life of vitamin B12 in the bloodstream in hours?", r"half-life"),
]


def main() -> None:
    from pathlib import Path

    chunks = load_chunks(Path(PDF))
    corpus = " ".join(c.text.lower() for c in chunks)
    by_page: dict[int, list[str]] = {}
    for c in chunks:
        by_page.setdefault(c.page, []).append(c.text)

    drafts = [json.loads(x) for x in (DATA / "golden_textbook_draft.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    kept = [d for d in drafts if d["id"] not in DROP and d["type"] != "unanswerable"]

    for d in kept:  # re-verify every evidence quote on its page
        d.pop("id")
        for ev, pg in zip(d["evidence"], d["pages"], strict=True):
            if not any(contains_evidence(t, ev) for t in by_page.get(pg, [])):
                raise SystemExit(f"evidence not found on page {pg}: {ev[:60]}")

    dropped_un = []
    for q, forbidden in UNANSWERABLE:
        if re.search(forbidden, corpus, re.IGNORECASE):
            dropped_un.append(q)
            continue
        kept.append({"type": "unanswerable", "question": q, "reference_answer": None, "evidence": [], "pages": []})

    out = [{"id": f"g{i + 1:03d}", **d} for i, d in enumerate(kept)]
    (DATA / "golden.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    counts = {t: sum(x["type"] == t for x in out) for t in ("single", "multi", "unanswerable")}
    print(f"dropped {len(DROP)} drafts; {len(dropped_un)} unanswerable candidates found in corpus (dropped)")
    print(f"-> {len(out)} questions {counts}")


if __name__ == "__main__":
    main()
