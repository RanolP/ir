#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["datasets"]
# ///
# Download mteb/Ko-StrategyQA and convert to BEIR JSONL format.
# Output: test-data/ko-strategyqa/{corpus,queries}.jsonl + qrels/test.tsv
# Usage: uv run scripts/download-ko-strategyqa.py

import json
import os
from datasets import load_dataset

OUT = os.path.join(os.path.dirname(__file__), "..", "test-data", "ko-strategyqa")
os.makedirs(os.path.join(OUT, "qrels"), exist_ok=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} rows → {path}")


print("loading corpus...")
corpus_ds = load_dataset("mteb/Ko-StrategyQA", "corpus", split="dev")
corpus = [
    {"_id": str(r["_id"]), "title": r.get("title", ""), "text": r.get("text", "")}
    for r in corpus_ds
]
write_jsonl(os.path.join(OUT, "corpus.jsonl"), corpus)

print("loading queries...")
queries_ds = load_dataset("mteb/Ko-StrategyQA", "queries", split="dev")
queries = [
    {"_id": str(r["_id"]), "text": r.get("text", "")}
    for r in queries_ds
]
write_jsonl(os.path.join(OUT, "queries.jsonl"), queries)

print("loading qrels...")
qrels_ds = load_dataset("mteb/Ko-StrategyQA", "qrels", split="dev")
qrels_path = os.path.join(OUT, "qrels", "test.tsv")
with open(qrels_path, "w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    count = 0
    for r in qrels_ds:
        f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")
        count += 1
print(f"  wrote {count:,} judgments → {qrels_path}")

print(f"\ndone. dataset at: {os.path.abspath(OUT)}")
