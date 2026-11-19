"""Compare retrieval strategies on the golden set (retrieval only, no LLM calls).
    python -m eval.ablate [--pdf ...]

Each row changes one thing so the effect is attributable. Gains are judged against the
bootstrap CIs in the full report, not just the point estimates here.
"""
import argparse
import time
from pathlib import Path

from app import llm
from eval.corpus import build_index
from eval.metrics import evidence_ranks, mean, recall_at_k, reciprocal_rank
from eval.run import DEFAULT_PDF, HERE, load_golden

CONFIGS = [
    ("vector (current)", {"mode": "vector"}),
    ("bm25 only", {"mode": "bm25"}),
    ("hybrid (vector+bm25, RRF)", {"mode": "hybrid"}),
    ("vector + rerank MiniLM-L6", {"mode": "vector", "reranker": "Xenova/ms-marco-MiniLM-L-6-v2"}),
    ("hybrid + rerank MiniLM-L6", {"mode": "hybrid", "reranker": "Xenova/ms-marco-MiniLM-L-6-v2"}),
    ("hybrid + rerank MiniLM-L12", {"mode": "hybrid", "reranker": "Xenova/ms-marco-MiniLM-L-12-v2"}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--golden", type=Path, default=HERE / "data" / "golden.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "ablation.md")
    args = ap.parse_args()

    index = build_index(args.pdf)
    items = [g for g in load_golden(args.golden) if g["evidence"]]
    qvecs = llm.embed_texts([g["question"] for g in items])
    rows = []
    for name, kw in CONFIGS:
        t0 = time.perf_counter()
        per = []
        for g, qv in zip(items, qvecs, strict=True):
            ranked = index.search(qv, 10, query_text=g["question"], candidates=30, **kw)
            per.append(evidence_ranks([r["content"] for r in ranked], g["evidence"]))
        ms = (time.perf_counter() - t0) / len(items) * 1000
        rows.append((name, [mean([recall_at_k(r, k) for r in per]) for k in (1, 3, 5, 10)], mean([reciprocal_rank(r) for r in per]), ms))
        print(f"{name:32s} R@1 {rows[-1][1][0]:.3f} R@5 {rows[-1][1][2]:.3f} MRR {rows[-1][2]:.3f} {ms:.0f}ms/q")

    lines = [f"# Retrieval ablation ({len(items)} answerable questions, {len(index.chunks)} chunks)", "",
             "| strategy | R@1 | R@3 | R@5 | R@10 | MRR | ms/query (excl. embed) |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {n} | " + " | ".join(f"{x:.3f}" for x in r) + f" | {m:.3f} | {ms:.0f} |" for n, r, m, ms in rows]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
