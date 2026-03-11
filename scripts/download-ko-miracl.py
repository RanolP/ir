#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["datasets<3.0"]
# ///
# Download MIRACL-Korean dev split and convert to BEIR JSONL format.
#
# Corpus is built from positive + negative passages embedded in query entries
# (no 1.5M corpus download needed). Negatives are hard negatives (BM25+DR retrieved),
# making the task harder and more realistic than random distractors.
#
# Scale: 213 queries, ~700-1000 passages (positives + hard negatives)
# Purpose: test if morphological preprocessors improve term recall on factoid Korean queries.
#
# Output: test-data/ko-miracl/{corpus,queries}.jsonl + qrels/test.tsv
# Usage: uv run scripts/download-ko-miracl.py

import json
import os

from datasets import load_dataset

OUT = os.path.join(os.path.dirname(__file__), "..", "test-data", "ko-miracl")
os.makedirs(os.path.join(OUT, "qrels"), exist_ok=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} rows → {path}")


print("loading MIRACL-Korean dev split...")
ds = load_dataset("miracl/miracl", "ko", split="dev", trust_remote_code=True)

queries = []
qrels = []
corpus: dict[str, dict] = {}  # docid → passage

for r in ds:
    qid = str(r["query_id"])
    queries.append({"_id": qid, "text": r["query"]})

    for p in r.get("positive_passages", []):
        pid = str(p["docid"])
        corpus[pid] = {"_id": pid, "title": p.get("title", ""), "text": p.get("text", "")}
        qrels.append((qid, pid, 1))

    for p in r.get("negative_passages", []):
        pid = str(p["docid"])
        corpus[pid] = {"_id": pid, "title": p.get("title", ""), "text": p.get("text", "")}
        # negatives get score=0 (not relevant)
        qrels.append((qid, pid, 0))

n_pos = sum(1 for _, _, s in qrels if s == 1)
n_neg = sum(1 for _, _, s in qrels if s == 0)
print(f"  {len(queries):,} queries, {len(corpus):,} passages ({n_pos} positive, {n_neg} hard-negative judgments)")

write_jsonl(os.path.join(OUT, "corpus.jsonl"), list(corpus.values()))
write_jsonl(os.path.join(OUT, "queries.jsonl"), queries)

qrels_path = os.path.join(OUT, "qrels", "test.tsv")
with open(qrels_path, "w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for qid, pid, score in qrels:
        f.write(f"{qid}\t{pid}\t{score}\n")
print(f"  wrote {len(qrels):,} judgments → {qrels_path}")

print(f"\ndone. dataset at: {os.path.abspath(OUT)}")
print(f"\nnote: corpus uses hard negatives (BM25+DR), not random distractors.")
print(f"nDCG@10 on this corpus is harder than random but smaller than full 1.5M Wikipedia.")
