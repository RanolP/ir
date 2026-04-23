#!/usr/bin/env python3
"""Prepare a self-contained smoltrain bundle for the tier-2 router."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_row(row: dict) -> dict:
    text = row.get("text") or row.get("input")
    if not text:
        raise ValueError("row missing both 'text' and 'input'")
    return {
        "text": text,
        "label": row["label"],
        "dataset": row.get("dataset", "unknown"),
        "lang": row.get("lang", "other"),
        "has_question": row.get("has_question", False),
        "query_id": row.get("query_id", ""),
        "query_text": row.get("query_text", ""),
    }


def stratified_split(rows: list[dict], eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[(row["dataset"], row["label"])].append(row)

    train: list[dict] = []
    eval_rows: list[dict] = []
    for items in by_bucket.values():
        bucket = items[:]
        rng.shuffle(bucket)
        n_eval = max(1, round(len(bucket) * eval_ratio))
        eval_rows.extend(bucket[:n_eval])
        train.extend(bucket[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_rows)
    return train, eval_rows


def balance_train_rows(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    target = max(len(items) for items in by_label.values())
    balanced: list[dict] = []
    for label, items in sorted(by_label.items()):
        balanced.extend(items)
        if len(items) < target:
            needed = target - len(items)
            balanced.extend(rng.choice(items) for _ in range(needed))

    rng.shuffle(balanced)
    return balanced


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_taxonomy(path: Path) -> None:
    path.write_text(
        """classes:
  skip_tier2:
    description: "Tier-1 fused result is strong enough; skip expansion and reranking."
  run_tier2:
    description: "Tier-1 signal is weak enough that tier-2 should run."

languages:
  - en
  - ko

config:
  latency_target_ms: 50
  f1_floor: 0.85
  agentic_recall_floor: 0.85
""",
        encoding="utf-8",
    )


def write_world(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "classes": {
                    "skip_tier2": {
                        "cross_class_discriminators": {
                            "vs_run_tier2": {
                                "key_signals": [
                                    "fused_top high",
                                    "fused_gap high",
                                    "bm25_gap high",
                                ]
                            }
                        }
                    },
                    "run_tier2": {
                        "cross_class_discriminators": {
                            "vs_skip_tier2": {
                                "key_signals": [
                                    "fused_top low",
                                    "fused_gap low",
                                    "bm25_gap low",
                                ]
                            }
                        }
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_readme(
    path: Path,
    input_path: Path,
    train_rows: list[dict],
    balanced_train_rows: list[dict],
    eval_rows: list[dict],
) -> None:
    counts = Counter(row["label"] for row in train_rows + eval_rows)
    balanced_counts = Counter(row["label"] for row in balanced_train_rows)
    path.write_text(
        "\n".join(
            [
                "Tier-2 router smoltrain bundle",
                "",
                f"source: {input_path}",
                f"train rows: {len(train_rows)}",
                f"train_balanced rows: {len(balanced_train_rows)}",
                f"eval rows: {len(eval_rows)}",
                f"labels: {dict(sorted(counts.items()))}",
                f"train_balanced labels: {dict(sorted(balanced_counts.items()))}",
                "",
                "Train (recommended, balanced):",
                "  cd THIS_DIR",
                "  PYTHONPATH=/Users/eliot/ws-ps/smoltrain python3 -m smoltrain.train \\",
                "    --data train_balanced.jsonl --taxonomy taxonomy.yaml --epochs 10 --seed 42",
                "",
                "Train (raw):",
                "  cd THIS_DIR",
                "  PYTHONPATH=/Users/eliot/ws-ps/smoltrain python3 -m smoltrain.train \\",
                "    --data train.jsonl --taxonomy taxonomy.yaml --epochs 10 --seed 42",
                "",
                "Eval:",
                "  cd THIS_DIR",
                "  PYTHONPATH=/Users/eliot/ws-ps/smoltrain python3 -m smoltrain.eval \\",
                "    --model models/charcnn_trained.onnx --data train.jsonl \\",
                "    --taxonomy taxonomy.yaml --world world.json --eval-data eval.jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare smoltrain bundle for the tier-2 router")
    ap.add_argument("--input", required=True, metavar="JSONL")
    ap.add_argument("--output-dir", required=True, metavar="DIR")
    ap.add_argument("--eval-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    rows = [normalize_row(row) for row in load_jsonl(input_path)]
    train_rows, eval_rows = stratified_split(rows, args.eval_ratio, args.seed)
    balanced_train_rows = balance_train_rows(train_rows, args.seed)

    write_jsonl(out_dir / "dataset.jsonl", rows)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "train_balanced.jsonl", balanced_train_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    write_taxonomy(out_dir / "taxonomy.yaml")
    write_world(out_dir / "world.json")
    write_readme(out_dir / "README.txt", input_path, train_rows, balanced_train_rows, eval_rows)

    print(f"prepared smoltrain bundle in {out_dir}")
    print(f"  dataset : {len(rows)}")
    print(f"  train   : {len(train_rows)}")
    print(f"  balanced: {len(balanced_train_rows)}")
    print(f"  eval    : {len(eval_rows)}")
    for label, count in sorted(Counter(row['label'] for row in rows).items()):
        print(f"  {label:>10}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
