#!/usr/bin/env python3
"""
pool-size-aggregate.py — Summarise pool-size variance study results.

Reads signal JSONL files from logs/signals/miracl-ko-s{size}-p{seed}/bm25.jsonl,
computes mean and between-seed stddev of nDCG@10 and Recall@100 per pool size,
and writes a recommendation table to research/pool-size-study.md.

Usage:
  python3 scripts/pool-size-aggregate.py \\
    --sizes 500,1000,2000,5000,10000 \\
    --seeds 5 \\
    --signals-root logs/signals \\
    --output research/pool-size-study.md
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load_metrics(signals_dir: Path, metric_keys: list[str]) -> dict[str, list[float]]:
    """Load per-query metrics from bm25.jsonl, return {metric: [values]}."""
    bm25_path = signals_dir / "bm25.jsonl"
    if not bm25_path.exists():
        return {}
    metrics: dict[str, list[float]] = {k: [] for k in metric_keys}
    with open(bm25_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for k in metric_keys:
                if k in rec:
                    metrics[k].append(float(rec[k]))
    return metrics


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def count_mandatory_docs(source_data: Path) -> int:
    qrels_path = source_data / "qrels" / "test.tsv"
    ids: set[str] = set()
    with open(qrels_path) as f:
        next(f)
        for line in f:
            _qid, doc_id, _score = line.rstrip("\n").split("\t")
            ids.add(doc_id)
    return len(ids)


def main():
    p = argparse.ArgumentParser(description="Aggregate pool-size variance study results")
    p.add_argument("--sizes", default="500,1000,2000,5000,10000")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--signals-root", default="logs/signals")
    p.add_argument("--source-data", default="test-data/miracl-ko")
    p.add_argument("--output", default="research/pool-size-study.md")
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = list(range(1, args.seeds + 1))
    signals_root = Path(args.signals_root)
    source_data = Path(args.source_data)
    mandatory_docs = count_mandatory_docs(source_data)
    metric_keys = ["ndcg10", "recall100"]

    rows = []
    for size in sizes:
        # Collect per-seed aggregate metrics (mean over queries)
        seed_ndcg = []
        seed_recall = []
        n_queries_list = []
        missing = []

        for seed in seeds:
            label = f"miracl-ko-s{size}-p{seed}"
            signals_dir = signals_root / label
            if not (signals_dir / ".done").exists():
                missing.append(seed)
                continue
            m = load_metrics(signals_dir, metric_keys)
            if not m.get("ndcg10"):
                missing.append(seed)
                continue
            seed_ndcg.append(mean(m["ndcg10"]))
            seed_recall.append(mean(m["recall100"]))
            n_queries_list.append(len(m["ndcg10"]))

        rows.append({
            "size": size,
            "n_seeds": len(seed_ndcg),
            "missing": missing,
            "ndcg_mean": mean(seed_ndcg),
            "ndcg_std": stddev(seed_ndcg),
            "recall_mean": mean(seed_recall),
            "recall_std": stddev(seed_recall),
            "n_queries": mean(n_queries_list) if n_queries_list else 0,
        })

    if not any(r["n_seeds"] > 0 for r in rows):
        print("ERROR: no completed runs found. Run scripts/pool-size-study.sh first.", file=sys.stderr)
        sys.exit(1)

    # Recommendation: smallest size where ndcg_std < 0.005
    THRESHOLD = 0.005
    recommended = None
    for r in rows:
        if r["size"] <= mandatory_docs:
            continue
        if r["n_seeds"] >= 3 and r["ndcg_std"] < THRESHOLD:
            recommended = r["size"]
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Pool-Size Variance Study — MIRACL-Ko BM25",
        "",
        "One-off study to find the minimum pool size that yields statistically stable",
        "nDCG@10 comparisons (between-seed stddev < 0.005).",
        "",
        f"Mandatory qrel-linked docs in source corpus: {mandatory_docs}.",
        "Any sampled size at or below that count is deterministic across seeds and",
        "must not be used for between-seed variance recommendations.",
        "",
        "## Results",
        "",
        "| Size | Seeds | nDCG@10 mean | nDCG@10 stddev | Recall@100 mean | Recall@100 stddev | Notes |",
        "|-----:|------:|-------------:|---------------:|----------------:|------------------:|-------|",
    ]
    for r in rows:
        if r["n_seeds"] == 0:
            note = f"missing (seeds: {r['missing']})"
            lines.append(f"| {r['size']:>5} | {r['n_seeds']:>5} | {'—':>12} | {'—':>14} | {'—':>15} | {'—':>17} | {note} |")
        else:
            note = ""
            if r["size"] <= mandatory_docs:
                note = f"deterministic floor (size <= mandatory_docs {mandatory_docs})"
            elif r["missing"]:
                note = f"partial (missing seeds {r['missing']})"
            elif r["size"] == recommended:
                note = "**recommended minimum**"
            flag = " ✓" if r["ndcg_std"] < THRESHOLD else ""
            lines.append(
                f"| {r['size']:>5} | {r['n_seeds']:>5} | {r['ndcg_mean']:>12.4f} | "
                f"{r['ndcg_std']:>14.4f}{flag} | {r['recall_mean']:>15.4f} | "
                f"{r['recall_std']:>17.4f} | {note} |"
            )

    lines += [
        "",
        "## Recommendation",
        "",
    ]
    if recommended:
        rec_row = next(r for r in rows if r["size"] == recommended)
        lines += [
            f"**Default pool size: {recommended} docs.**",
            "",
            f"At size {recommended}, between-seed nDCG@10 stddev falls below the 0.005 threshold "
            f"(measured {rec_row['ndcg_std']:.4f}), meaning pool-to-pool variance is smaller than "
            "typical improvement signals. Smaller sizes may work for stability/speed-only checks but "
            "are not reliable for detecting metric regressions, and deterministic undersized pools "
            "must be excluded from this decision.",
            "",
            f"Update the `/benchmark` skill's Phase 1 corpus-check table: `miracl-ko --size {recommended}`.",
        ]
    else:
        lines += [
            "**No size below the 0.005 threshold found.** Options:",
            "",
            "1. Increase seed count (--seeds 10) for better estimates at large sizes.",
            "2. Raise threshold to 0.010 and accept more variance.",
            "3. Use size 10000 as the working default and re-run this study when the corpus changes.",
        ]

    lines += [
        "",
        f"*Generated by `scripts/pool-size-aggregate.py`*",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written: {out_path}")

    # Print summary
    print("\nSummary:")
    for r in rows:
        if r["n_seeds"] > 0:
            flag = " ← recommended" if r["size"] == recommended else ""
            print(f"  size={r['size']:>6}  seeds={r['n_seeds']}  "
                  f"ndcg_std={r['ndcg_std']:.4f}{flag}")


if __name__ == "__main__":
    main()
