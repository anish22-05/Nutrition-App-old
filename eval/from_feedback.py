"""Turn real users' rated answers into draft evaluation cases.

    python -m eval.from_feedback [--out eval/data/feedback_draft.jsonl]

The golden set was written by an LLM from the book itself, which flatters retrieval (the
questions reuse the book's wording). Real questions are the honest test. This exports every
rated interaction - thumbs-down first, since those are the failures worth fixing - in the golden
set's format, with `needs_review: true`: a person still has to write the reference answer and
pick the evidence sentence before a case joins eval/data/golden.jsonl.
"""
import argparse
import json
from pathlib import Path

from app import db


def to_case(row: dict, n: int) -> dict:
    pages = sorted({s.get("page") for s in row.get("sources") or [] if s.get("page") is not None})
    return {
        "id": f"fb{n:03d}",
        "type": "from_feedback",
        "question": row.get("standalone") or row["question"],
        "reference_answer": row["answer"] if row.get("rating") == 1 else "",
        "evidence": [],
        "pages": pages,
        "needs_review": True,
        "user_rating": row.get("rating"),
        "user_comment": row.get("comment"),
        "answer_given": row["answer"],
        "request_id": row["request_id"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "feedback_draft.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    rows = sorted(db.list_interactions(rated_only=True, limit=args.limit), key=lambda r: (r.get("rating") or 0, r["created_at"]))
    cases = [to_case(r, i) for i, r in enumerate(rows, start=1)]
    args.out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    down = sum(1 for c in cases if c["user_rating"] == -1)
    print(f"{len(cases)} rated answers ({down} thumbs-down) -> {args.out}")


if __name__ == "__main__":
    main()
