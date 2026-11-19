"""Turn golden_draft.jsonl into golden.jsonl.   python -m eval.curate

This is the human-in-the-loop step, written down so it is reproducible and reviewable:
  * DROP: draft questions removed after reading them, each with the reason.
  * HAND: questions written by hand to *paraphrase* the slides. LLM-drafted questions tend to
    echo the source wording, which flatters embedding retrieval; these do not.
Every evidence quote (drafted or hand-written) is re-verified against the corpus.
"""
import json

from eval.corpus import load_chunks
from eval.generate_dataset import DATA, DEFAULT_PDF
from eval.metrics import contains_evidence

DROP = {
    "q001": "duplicate of q002 (two slides list complex-carb sources); other page is an equally valid answer",
    "q018": "three slides describe functions of water; a hit on another page would be scored a false miss",
    "q020": "same as q018",
    "q023": "overlaps q032/q040 (why the body needs food / nutrient functions / functions of food)",
    "q032": "same as q023",
    "q030": "question only makes sense with the riddle-style slide; ambiguous on its own",
    "q046": "multi-chunk question too vague to have one correct answer",
    "q049": "multi-chunk question contrived; evidence does not need both passages",
    "q050": "multi-chunk question contrived (random pairing)",
    "q052": "multi-chunk question muddled ('micronutrients like vitamins and minerals')",
    "q054": "multi-chunk question contrived (random pairing)",
    "q055": "multi-chunk question too vague",
}

# (question, reference answer, verbatim evidence, page)
HAND = [
    (
        "Which vitamin can improve how much iron the body absorbs from a meal?",
        "Vitamin C, when foods or drinks containing both vitamin C and iron are consumed at the same meal.",
        "Vitamin C can help with the absorption of iron when foods or drink containing both vitamin C and iron are eaten at the same meal.",
        35,
    ),
    (
        "Why might teenage girls need to pay extra attention to what they eat?",
        "Women and teenage girls have higher iron requirements than men of the same age.",
        "Women and teenage girls need to ensure they have enough because their requirements are higher than those of men of the same age",
        42,
    ),
    (
        "What is the most widespread nutritional deficiency in the world?",
        "Iron deficiency (anaemia), affecting more than 2 billion people worldwide.",
        "More than 2 billion people worldwide suffer from iron deficiency anaemia, making it the most common nutritional deficiency.",
        42,
    ),
    (
        "Which dietary mineral is linked to hypertension?",
        "Sodium: high sodium intake is a risk factor for high blood pressure, which may lead to heart disease and stroke.",
        "High sodium intake is considered to be one of the risk factors for high blood pressure, which may lead to heart disease and stroke.",
        44,
    ),
    (
        "Which mineral does the body need to build the oxygen-carrying part of the blood?",
        "Iron, which is essential for forming haemoglobin in red blood cells.",
        "Iron is essential for the formation of haemoglobin in red blood cells.",
        40,
    ),
    (
        "Apart from bones and teeth, what else depends on the most abundant mineral in the body?",
        "Calcium is also needed for blood clotting and normal muscle function.",
        "It is essential for a number of important functions such as the maintenance of bones and teeth, blood clotting and normal muscle function.",
        39,
    ),
    (
        "In what ways does the body lose water during the day?",
        "Through going to the toilet, sweating, and evaporation from the skin.",
        "The body loses water all the time, when we go to the toilet, from sweat and also by evaporation from skin.",
        48,
    ),
]


def main() -> None:
    chunks = load_chunks(DEFAULT_PDF)
    by_page: dict[int, list[str]] = {}
    for c in chunks:
        by_page.setdefault(c.page, []).append(c.text)

    drafts = [json.loads(line) for line in (DATA / "golden_draft.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [d for d in drafts if d["id"] not in DROP]
    for d in kept:
        d.pop("id")

    for q, a, ev, page in HAND:
        if not any(contains_evidence(t, ev) for t in by_page.get(page, [])):
            raise SystemExit(f"hand-written evidence not found on page {page}: {ev[:60]}")
        kept.append({"type": "single", "question": q, "reference_answer": a, "evidence": [ev], "pages": [page], "source": "hand"})

    out = [{"id": f"g{i + 1:03d}", **d} for i, d in enumerate(kept)]
    (DATA / "golden_deck.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    counts = {t: sum(x["type"] == t for x in out) for t in ("single", "multi", "unanswerable")}
    print(f"dropped {len(DROP)} drafts, added {len(HAND)} hand-written -> {len(out)} questions {counts}")


if __name__ == "__main__":
    main()
