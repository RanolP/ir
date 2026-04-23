#!/usr/bin/env python3
"""Offline holdout benchmark for the tier-2 router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from router_features import feature_text, tier2_label
from smoltrain.model import CharCNN, encode_text

WARMUP = 5


def load_jsonl(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            by_id[row["query_id"]] = row
    return rows, by_id


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


def aggregate(rows: list[dict]) -> dict:
    latencies = [float(r["elapsed_ms"]) for r in rows[WARMUP:]]
    return {
        "n_queries": len(rows),
        "ndcg10": sum(float(r.get("ndcg10", 0.0)) for r in rows) / len(rows),
        "recall10": sum(float(r.get("recall10", 0.0)) for r in rows) / len(rows),
        "median_ms": percentile(latencies, 50),
    }


def load_model(checkpoint: Path) -> tuple[CharCNN, dict[str, int]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    model = CharCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, {label: idx for idx, label in enumerate(classes)}


@torch.no_grad()
def run_probability(model: CharCNN, class_to_idx: dict[str, int], text: str) -> float:
    x = torch.tensor(encode_text(text), dtype=torch.long).unsqueeze(0)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    return float(probs[class_to_idx["run_tier2"]])


def evaluate_threshold(
    threshold: float,
    query_order: list[str],
    bm25: dict[str, dict],
    vector: dict[str, dict],
    skip_rows: dict[str, dict],
    hybrid: dict[str, dict],
    model: CharCNN,
    class_to_idx: dict[str, int],
    ndcg_margin: float,
    dataset: str,
) -> dict:
    chosen_rows: list[dict] = []
    tp = fp = tn = fn = 0
    run_count = 0

    for qid in query_order:
        b = bm25[qid]
        v = vector[qid]
        s = skip_rows[qid]
        h = hybrid[qid]
        text = feature_text(dataset, h["query_text"], b, v, h)
        p_run = run_probability(model, class_to_idx, text)
        pred_run = p_run >= threshold
        true_run = tier2_label(s, h, ndcg_margin) == "run_tier2"

        if pred_run:
            run_count += 1
            chosen_rows.append(h)
        else:
            chosen_rows.append(s)

        if pred_run and true_run:
            tp += 1
        elif pred_run and not true_run:
            fp += 1
        elif not pred_run and true_run:
            fn += 1
        else:
            tn += 1

    agg = aggregate(chosen_rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    agg.update(
        {
            "policy": f"router-{threshold:.3f}",
            "run_rate": run_count / len(query_order),
            "precision": precision,
            "recall": recall,
        }
    )
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark a tier-2 router checkpoint on holdout signals")
    ap.add_argument("--signals", required=True, metavar="DIR")
    ap.add_argument("--checkpoint", required=True, metavar="PT")
    ap.add_argument("--thresholds", default="0.5", help="Comma-separated run_tier2 probability thresholds")
    ap.add_argument("--skip-source", choices=["tier1", "vector"], default="tier1")
    ap.add_argument("--ndcg-margin", type=float, default=0.02)
    ap.add_argument("--output", metavar="JSON")
    args = ap.parse_args()

    signals_dir = Path(args.signals)
    checkpoint = Path(args.checkpoint)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    hybrid_rows, hybrid = load_jsonl(signals_dir / "hybrid.jsonl")
    _, bm25 = load_jsonl(signals_dir / "bm25.jsonl")
    _, vector = load_jsonl(signals_dir / "vector.jsonl")
    _, skip_rows = load_jsonl(signals_dir / f"{args.skip_source}.jsonl")

    query_order = [row["query_id"] for row in hybrid_rows]
    dataset = signals_dir.name
    model, class_to_idx = load_model(checkpoint)

    skip_policy = aggregate([skip_rows[qid] for qid in query_order])
    skip_policy.update({"policy": args.skip_source, "run_rate": 0.0, "precision": 0.0, "recall": 0.0})

    hybrid_policy = aggregate([hybrid[qid] for qid in query_order])
    oracle_rate = sum(
        1
        for qid in query_order
        if tier2_label(skip_rows[qid], hybrid[qid], args.ndcg_margin) == "run_tier2"
    ) / len(query_order)
    hybrid_policy.update({"policy": "hybrid", "run_rate": 1.0, "precision": oracle_rate, "recall": 1.0})

    rows = [skip_policy]
    rows.extend(
        evaluate_threshold(
            threshold,
            query_order,
            bm25,
            vector,
            skip_rows,
            hybrid,
            model,
            class_to_idx,
            args.ndcg_margin,
            dataset,
        )
        for threshold in thresholds
    )
    rows.append(hybrid_policy)

    hybrid_ndcg = hybrid_policy["ndcg10"]
    hybrid_recall = hybrid_policy["recall10"]
    hybrid_median = hybrid_policy["median_ms"]

    print("\nRouter holdout benchmark")
    print("=" * 110)
    print(
        f"{'policy':<14} {'nDCG@10':>8} {'ΔnDCG':>8} {'R@10':>8} {'ΔR@10':>8} "
        f"{'med ms':>8} {'Δmed%':>8} {'run%':>8} {'prec':>8} {'recall':>8}"
    )
    print(
        f"{'-' * 14} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} "
        f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}"
    )
    for row in rows:
        delta_ndcg = row["ndcg10"] - hybrid_ndcg
        delta_recall = row["recall10"] - hybrid_recall
        delta_median = ((row["median_ms"] - hybrid_median) / hybrid_median * 100) if hybrid_median else 0.0
        print(
            f"{row['policy']:<14} "
            f"{row['ndcg10']:>8.4f} {delta_ndcg:>8.4f} "
            f"{row['recall10']:>8.4f} {delta_recall:>8.4f} "
            f"{row['median_ms']:>8.1f} {delta_median:>8.1f} "
            f"{row['run_rate'] * 100:>7.1f}% "
            f"{row['precision']:>8.4f} {row['recall']:>8.4f}"
        )

    if args.output:
        payload = {
            "signals": str(signals_dir),
            "checkpoint": str(checkpoint),
            "skip_source": args.skip_source,
            "thresholds": thresholds,
            "rows": rows,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
