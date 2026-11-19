"""Run the evaluation.   python -m eval.run [--no-generation] [--save-baseline] [--check]

Retrieval is scored on the ranked top-10 with no similarity cut-off (pure ranking quality).
Generation is scored the way the API actually behaves: top-k = DEFAULT_TOP_K and the
MIN_SIMILARITY gate applied, then rag.answer() - the same function /chat calls.
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from app import llm, rag
from app.config import get_settings
from eval import judge
from eval.corpus import build_index, plan_all, search_planned
from eval.metrics import (
    auc,
    bootstrap_ci,
    evidence_ranks,
    mean,
    recall_at_k,
    reciprocal_rank,
)

HERE = Path(__file__).parent
DEFAULT_PDF = Path("data/Human-Nutrition-2020.pdf")
KS = (1, 3, 5, 10)
THRESHOLDS = (0.2, 0.3, 0.35, 0.4, 0.5, 0.6)
# Metrics where a *drop* is a regression (false_refusal_rate is the reverse and handled below).
HIGHER_IS_BETTER = ("recall@5", "mrr", "correctness", "faithfulness", "abstention_accuracy")


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stat(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    lo, hi = bootstrap_ci(vals)
    return {"mean": round(mean(vals), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}


def evaluate_item(item: dict, qvec: list[float], index, generate: bool, judge_model: str, search_kw: dict | None = None,
                  plan: tuple[list[str], list] | None = None) -> dict:
    """`plan` = (sub-queries, their vectors) when query decomposition is on; else the question itself."""
    s = get_settings()
    queries, vecs = plan or ([item["question"]], [qvec])
    kw = dict(search_kw or {})
    ranked = search_planned(index, queries, vecs, max(KS), **kw)
    rec: dict = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "top1_similarity": ranked[0]["similarity"] if ranked else 0.0,
        "top_pages": [r["metadata"]["page"] for r in ranked[:5]],
    }

    if item["evidence"]:
        ranks = evidence_ranks([r["content"] for r in ranked], item["evidence"])
        rec["ranks"] = ranks
        rec["gold_pages"] = item["pages"]
        rec["reciprocal_rank"] = reciprocal_rank(ranks)
        for k in KS:
            rec[f"recall@{k}"] = recall_at_k(ranks, k)

    if generate:
        matches = search_planned(index, queries, vecs, s.default_top_k, s.min_similarity, **kw)
        t0 = time.perf_counter()
        answer = rag.answer(item["question"], matches)
        rec["latency_s"] = round(time.perf_counter() - t0, 2)
        rec["answer"] = answer
        rec["abstained"] = judge.is_abstention(answer)
        rec["gated_by_similarity"] = not matches

        if item["type"] != "unanswerable" and not rec["abstained"]:
            rec["faithfulness"] = judge.faithfulness(answer, matches, judge_model)
            rec["correctness"], rec["judge_reason"] = judge.correctness(
                item["question"], answer, item["reference_answer"], judge_model
            )
            rec["citations_valid"] = judge.valid_citations(answer, len(matches))
        elif item["type"] != "unanswerable":
            rec["correctness"] = 0.0  # a false refusal is a wrong answer
    return rec


def summarize(records: list[dict], generate: bool) -> dict:
    answerable = [r for r in records if r["type"] != "unanswerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    out: dict = {"n_questions": len(records), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}

    out["retrieval"] = {f"recall@{k}": stat([r[f"recall@{k}"] for r in answerable]) for k in KS}
    out["retrieval"]["mrr"] = stat([r["reciprocal_rank"] for r in answerable])
    for t in ("single", "multi"):
        sub = [r for r in answerable if r["type"] == t]
        if sub:
            out["retrieval"][f"recall@5[{t}]"] = stat([r["recall@5"] for r in sub])

    ans_sims = [r["top1_similarity"] for r in answerable]
    un_sims = [r["top1_similarity"] for r in unanswerable]
    out["similarity_gate"] = {
        "auc_answerable_vs_unanswerable": round(auc(ans_sims, un_sims), 4),
        "mean_top1_answerable": round(mean(ans_sims), 4),
        "mean_top1_unanswerable": round(mean(un_sims), 4),
        "sweep": [
            {
                "threshold": t,
                "answerable_kept": round(mean([float(x >= t) for x in ans_sims]), 3),
                "unanswerable_blocked": round(mean([float(x < t) for x in un_sims]), 3),
            }
            for t in THRESHOLDS
        ],
    }

    if generate:
        got = [r for r in answerable if not r["abstained"]]
        gen = {
            "correctness": stat([r["correctness"] for r in answerable]),
            "faithfulness": stat([r.get("faithfulness") for r in got]),
            "citation_valid_rate": stat([float(r["citations_valid"]) for r in got]),
            "false_refusal_rate": stat([float(r["abstained"]) for r in answerable]),
            "abstention_accuracy": stat([float(r["abstained"]) for r in unanswerable]),
            "unanswerable_blocked_by_gate": stat([float(r["gated_by_similarity"]) for r in unanswerable]),
            "latency_s_mean": round(mean([r["latency_s"] for r in records]), 2),
        }
        out["generation"] = gen
        out["headline"] = {
            "recall@5": out["retrieval"]["recall@5"]["mean"],
            "mrr": out["retrieval"]["mrr"]["mean"],
            "correctness": gen["correctness"]["mean"],
            "faithfulness": gen["faithfulness"]["mean"],
            "abstention_accuracy": gen["abstention_accuracy"]["mean"],
            "false_refusal_rate": gen["false_refusal_rate"]["mean"],
        }
    else:
        out["headline"] = {"recall@5": out["retrieval"]["recall@5"]["mean"], "mrr": out["retrieval"]["mrr"]["mean"]}
    return out


def git_state() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE, check=False).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE, check=False).stdout.strip()
        return f"{sha}{'+dirty' if dirty else ''}"
    except OSError:
        return "unknown"


def fmt(s: dict) -> str:
    return f"{s['mean']:.3f}  [{s['ci95'][0]:.2f}-{s['ci95'][1]:.2f}]"


def write_report(summary: dict, records: list[dict], meta: dict, path: Path) -> None:
    r = summary["retrieval"]
    lines = [
        "# Evaluation report",
        "",
        (
            f"`{meta['timestamp']}` · commit `{meta['git']}` · corpus `{meta['pdf']}` ({meta['n_chunks']} chunks) · "
            f"embed `{meta['embed_model']}` · chat `{meta['chat_model']}` · judge `{meta['judge_model']}`"
        ),
        (
            f"{summary['n_answerable']} answerable + {summary['n_unanswerable']} unanswerable questions. "
            "Values are mean [95% bootstrap CI]."
        ),
        "",
        "## Retrieval (answerable questions, no similarity cut-off)",
        "",
        "| metric | score |",
        "|---|---|",
        *[f"| Recall@{k} | {fmt(r[f'recall@{k}'])} |" for k in KS],
        f"| MRR@10 | {fmt(r['mrr'])} |",
    ]
    for t in ("single", "multi"):
        if f"recall@5[{t}]" in r:
            lines.append(f"| Recall@5 ({t}-chunk questions, n={r[f'recall@5[{t}]']['n']}) | {fmt(r[f'recall@5[{t}]'])} |")

    if "generation" in summary:
        g = summary["generation"]
        lines += [
            "",
            f"## Generation (top-{meta['top_k']}, similarity gate {meta['min_similarity']})",
            "",
            "| metric | score |",
            "|---|---|",
            f"| Answer correctness (vs reference, partial = 0.5) | {fmt(g['correctness'])} |",
            f"| Faithfulness (claims supported by context) | {fmt(g['faithfulness'])} |",
            f"| Citation validity (answers citing real sources) | {fmt(g['citation_valid_rate'])} |",
            f"| False-refusal rate (answerable but refused; lower is better) | {fmt(g['false_refusal_rate'])} |",
            f"| Abstention accuracy (unanswerable, refused correctly) | {fmt(g['abstention_accuracy'])} |",
            f"| Mean latency per answer | {g['latency_s_mean']} s |",
        ]

    gate = summary["similarity_gate"]
    lines += [
        "",
        "## Can similarity alone tell answerable from unanswerable?",
        "",
        (
            f"AUC of top-1 similarity = **{gate['auc_answerable_vs_unanswerable']}** "
            f"(mean {gate['mean_top1_answerable']} answerable vs {gate['mean_top1_unanswerable']} unanswerable)."
        ),
        "",
        "| threshold | answerable kept | unanswerable blocked |",
        "|---|---|---|",
        *[f"| {x['threshold']} | {x['answerable_kept']:.0%} | {x['unanswerable_blocked']:.0%} |" for x in gate["sweep"]],
    ]

    misses = [x for x in records if x["type"] != "unanswerable" and x["recall@5"] < 1]
    lines += ["", f"## Retrieval misses at k=5 ({len(misses)})", ""]
    for x in misses:
        lines.append(f"- **{x['id']}** {x['question']}  \n  gold pages {x['gold_pages']} · retrieved pages {x['top_pages']} · evidence ranks {x['ranks']}")

    if "generation" in summary:
        bad = [x for x in records if x["type"] != "unanswerable" and x.get("correctness", 1) < 1]
        lines += ["", f"## Answers not fully correct ({len(bad)})", ""]
        for x in bad:
            why = x.get("judge_reason") or ("refused to answer" if x.get("abstained") else "")
            lines.append(f"- **{x['id']}** {x['question']}  \n  {why}")
        wrong_un = [x for x in records if x["type"] == "unanswerable" and not x["abstained"]]
        lines += ["", f"## Unanswerable questions that got an answer instead of a refusal ({len(wrong_un)})", ""]
        for x in wrong_un:
            lines.append(f"- **{x['id']}** {x['question']}  \n  -> {x['answer'][:200]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_regression(headline: dict, baseline_path: Path, tolerance: float) -> list[str]:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))["headline"]
    problems = []
    for k in HIGHER_IS_BETTER:
        if k in headline and k in base and headline[k] < base[k] - tolerance:
            problems.append(f"{k}: {headline[k]:.3f} < baseline {base[k]:.3f} (tolerance {tolerance})")
    if "false_refusal_rate" in headline and headline["false_refusal_rate"] > base["false_refusal_rate"] + tolerance:
        problems.append(f"false_refusal_rate: {headline['false_refusal_rate']:.3f} > baseline {base['false_refusal_rate']:.3f}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--golden", type=Path, default=HERE / "data" / "golden.jsonl")
    ap.add_argument("--no-generation", action="store_true", help="retrieval metrics only (cheap, fast)")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 1 if the headline metrics regress vs baseline.json")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default=None, help="first-stage retrieval (default: app setting)")
    ap.add_argument("--rerank", choices=["on", "off"], default=None, help="cross-encoder rerank (default: app setting)")
    ap.add_argument("--contextual", choices=["on", "off"], default=None, help="section-aware embeddings (default: app setting)")
    ap.add_argument("--decompose", choices=["on", "off"], default=None, help="split multi-part questions (default: app setting)")
    ap.add_argument("--out-name", default="latest", help="report file name under eval/results/")
    args = ap.parse_args()

    s = get_settings()
    generate = not args.no_generation
    contextual = s.contextual_embeddings if args.contextual is None else args.contextual == "on"
    decompose = s.query_decomposition if args.decompose is None else args.decompose == "on"
    index = build_index(args.pdf, contextual=contextual)
    search_kw = {
        "mode": args.mode or s.retrieval_mode,
        "candidates": s.rerank_candidates,
        "reranker": (s.reranker_model if (args.rerank or ('on' if s.rerank_enabled else 'off')) == 'on' else None),
    }
    golden = load_golden(args.golden)
    qvecs = llm.embed_texts([g["question"] for g in golden])
    plans: dict[str, tuple[list[str], list]] = {}
    if decompose:
        planned = plan_all([g["question"] for g in golden], args.workers)
        multi = {q: p for q, p in planned.items() if p != [q]}
        if multi:
            flat = [x for p in multi.values() for x in p]
            vec_of = dict(zip(flat, llm.embed_texts(flat), strict=True))
            plans = {q: (p, [vec_of[x] for x in p]) for q, p in multi.items()}
    print(f"{len(golden)} questions · {len(index.chunks)} chunks · generation={'on' if generate else 'off'} · "
          f"contextual={contextual} · decomposed={len(plans)} questions")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(
            lambda a: evaluate_item(a[0], a[1], index, generate, s.judge_model, search_kw, plans.get(a[0]["question"])),
            zip(golden, qvecs, strict=True),
        ))

    summary = summarize(records, generate)
    meta = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git": git_state(),
        "pdf": args.pdf.name,
        "n_chunks": len(index.chunks),
        "embed_model": s.embed_model,
        "chat_model": s.chat_model,
        "judge_model": s.judge_model,
        "top_k": s.default_top_k,
        "min_similarity": s.min_similarity,
        "retrieval": {"mode": search_kw["mode"], "reranker": search_kw["reranker"]},
        "chunking": {"sents_per_chunk": s.sents_per_chunk, "overlap": s.sent_overlap},
        "contextual_embeddings": contextual,
        "query_decomposition": decompose,
    }

    results = HERE / "results"
    results.mkdir(exist_ok=True)
    (results / f"{args.out_name}.json").write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, records, meta, results / f"{args.out_name}.md")
    print(json.dumps(summary["headline"], indent=2))
    print(f"report: {results / (args.out_name + '.md')}")

    baseline = HERE / "baseline.json"
    if args.save_baseline:
        baseline.write_text(json.dumps({"meta": meta, "headline": summary["headline"]}, indent=2), encoding="utf-8")
        print(f"baseline saved to {baseline}")
    if args.check:
        problems = check_regression(summary["headline"], baseline, args.tolerance)
        if problems:
            print("REGRESSION:\n  " + "\n  ".join(problems))
            raise SystemExit(1)
        print("no regression vs baseline")


if __name__ == "__main__":
    main()
