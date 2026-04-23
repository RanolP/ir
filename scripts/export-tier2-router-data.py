#!/usr/bin/env python3
"""Build tier-2 routing JSONL from collected signal runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from router_features import detect_lang, feature_text, has_question, tier2_label


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["query_id"]] = row
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Export tier-2 routing data for smoltrain")
    ap.add_argument("signals_dirs", nargs="+", metavar="DIR")
    ap.add_argument("--output", required=True, metavar="JSONL")
    ap.add_argument("--ndcg-margin", type=float, default=0.02)
    ap.add_argument(
        "--skip-source",
        choices=["tier1", "vector"],
        default="tier1",
        help="Baseline used for the skip decision label (default: tier1)",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"run_tier2": 0, "skip_tier2": 0}
    rows: list[dict[str, str]] = []

    for raw_dir in args.signals_dirs:
        signals_dir = Path(raw_dir)
        dataset = signals_dir.name
        bm25 = load_jsonl(signals_dir / "bm25.jsonl")
        vector = load_jsonl(signals_dir / "vector.jsonl")
        hybrid = load_jsonl(signals_dir / "hybrid.jsonl")
        skip_rows = load_jsonl(signals_dir / f"{args.skip_source}.jsonl")

        for query_id, h in hybrid.items():
            v = vector.get(query_id)
            b = bm25.get(query_id)
            skip = skip_rows.get(query_id)
            if not v or not b or not skip:
                continue

            label = tier2_label(skip, h, args.ndcg_margin)
            counts[label] += 1
            rows.append(
                {
                    "input": feature_text(dataset, h["query_text"], b, v, h),
                    "text": feature_text(dataset, h["query_text"], b, v, h),
                    "label": label,
                    "dataset": dataset,
                    "lang": detect_lang(h["query_text"]),
                    "has_question": has_question(h["query_text"]),
                    "query_id": query_id,
                    "query_text": h["query_text"],
                    "skip_source": args.skip_source,
                }
            )

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {out_path}")
    print(f"  run_tier2 : {counts['run_tier2']}")
    print(f"  skip_tier2: {counts['skip_tier2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
