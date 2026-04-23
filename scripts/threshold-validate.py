#!/usr/bin/env python3
"""
threshold-validate.py — shortlist stable threshold candidates from sweep JSON.

Input is the JSON emitted by scripts/threshold-sweep.py. Output is a ranked
candidate list suitable for holdout validation.
"""

import argparse
import json
from pathlib import Path


def load_sweep(path: Path) -> list[dict]:
    return json.load(open(path))


def number_token(value: float) -> str:
    return format(value, ".15g").replace("-", "m").replace(".", "_")


def aggregate_candidates(
    datasets: list[dict], gate: str, max_harm: float, min_fire: float
) -> list[dict]:
    if gate == "fused":
        grouped: dict[float, list[dict]] = {}
        for dataset in datasets:
            for row in dataset.get("fused_sweep", []):
                grouped.setdefault(float(row["product"]), []).append(
                    {
                        "label": dataset["label"],
                        "fire_rate": float(row["fire_rate"]),
                        "harm_rate": float(row["harm_rate"]),
                        "mean_loss": float(row.get("mean_loss", 0.0)),
                    }
                )

        candidates = []
        for product, rows in grouped.items():
            if len(rows) != len(datasets):
                continue
            passes = all(
                row["harm_rate"] <= max_harm and row["fire_rate"] >= min_fire for row in rows
            )
            candidates.append(
                {
                    "gate": gate,
                    "product": product,
                    "label": f"product-{number_token(product)}",
                    "labels": [row["label"] for row in rows],
                    "mean_fire_rate": sum(row["fire_rate"] for row in rows) / len(rows),
                    "max_harm_rate": max(row["harm_rate"] for row in rows),
                    "mean_loss": sum(row["mean_loss"] for row in rows) / len(rows),
                    "passes": passes,
                    "per_dataset": rows,
                }
            )
        candidates.sort(
            key=lambda row: (
                not row["passes"],
                -row["mean_fire_rate"],
                row["max_harm_rate"],
                abs(row["mean_loss"]),
                row["product"],
            )
        )
        return candidates

    grouped: dict[tuple[float, float], list[dict]] = {}
    for dataset in datasets:
        for row in dataset.get("bm25_sweep", []):
            key = (float(row["floor"]), float(row["gap"]))
            grouped.setdefault(key, []).append(
                {
                    "label": dataset["label"],
                    "fire_rate": float(row["fire_rate"]),
                    "harm_rate": float(row["harm_rate"]),
                    "mean_loss": float(row.get("mean_loss", 0.0)),
                }
            )

    candidates = []
    for (floor, gap), rows in grouped.items():
        if len(rows) != len(datasets):
            continue
        passes = all(
            row["harm_rate"] <= max_harm and row["fire_rate"] >= min_fire for row in rows
        )
        candidates.append(
            {
                "gate": gate,
                "floor": floor,
                "gap": gap,
                "label": f"floor-{number_token(floor)}-gap-{number_token(gap)}",
                "labels": [row["label"] for row in rows],
                "mean_fire_rate": sum(row["fire_rate"] for row in rows) / len(rows),
                "max_harm_rate": max(row["harm_rate"] for row in rows),
                "mean_loss": sum(row["mean_loss"] for row in rows) / len(rows),
                "passes": passes,
                "per_dataset": rows,
            }
        )
    candidates.sort(
        key=lambda row: (
            not row["passes"],
            -row["mean_fire_rate"],
            row["max_harm_rate"],
            abs(row["mean_loss"]),
            row["floor"],
            row["gap"],
        )
    )
    return candidates


def print_candidates(candidates: list[dict], gate: str, top: int):
    title = "Tier-1 fused" if gate == "fused" else "Tier-0 BM25"
    print(f"{title} shortlist")
    print("=" * 72)
    if gate == "fused":
        print(f"  {'product':>7}  {'pass':>4}  {'fire%':>6}  {'max harm%':>9}  {'avg loss':>9}")
        for row in candidates[:top]:
            print(
                f"  {row['product']:>7.3f}  "
                f"{'yes' if row['passes'] else 'no':>4}  "
                f"{row['mean_fire_rate']*100:>5.1f}%  "
                f"{row['max_harm_rate']*100:>8.1f}%  "
                f"{row['mean_loss']:>9.4f}"
            )
    else:
        print(f"  {'floor':>5}  {'gap':>5}  {'pass':>4}  {'fire%':>6}  {'max harm%':>9}  {'avg loss':>9}")
        for row in candidates[:top]:
            print(
                f"  {row['floor']:>5.2f}  {row['gap']:>5.2f}  "
                f"{'yes' if row['passes'] else 'no':>4}  "
                f"{row['mean_fire_rate']*100:>5.1f}%  "
                f"{row['max_harm_rate']*100:>8.1f}%  "
                f"{row['mean_loss']:>9.4f}"
            )


def main():
    ap = argparse.ArgumentParser(description="Shortlist threshold candidates from sweep JSON")
    ap.add_argument("sweep_json", help="JSON output from scripts/threshold-sweep.py")
    ap.add_argument("--gate", choices=["fused", "bm25"], default="fused")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--max-harm", type=float, default=0.05)
    ap.add_argument("--min-fire", type=float, default=0.10)
    ap.add_argument("--output", "-o", help="Write shortlisted candidates to file")
    args = ap.parse_args()

    datasets = load_sweep(Path(args.sweep_json))
    candidates = aggregate_candidates(datasets, args.gate, args.max_harm, args.min_fire)
    print_candidates(candidates, args.gate, args.top)

    shortlisted = [row for row in candidates if row["passes"]][: args.top]
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(shortlisted, f, indent=2)
        print(f"\nShortlist written to {args.output}")


if __name__ == "__main__":
    main()
