#!/usr/bin/env bash
# download-miracl-ko.sh — Download MIRACL Korean corpus in BEIR format
# Output: test-data/miracl-ko/{corpus.jsonl, queries.jsonl, qrels/test.tsv}
#
# Requires: python3, pip install datasets huggingface_hub
set -euo pipefail

OUT="test-data/miracl-ko"
mkdir -p "$OUT/qrels"

python3 - <<'PYEOF'
import json, os, pathlib, sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: run: pip install datasets", file=sys.stderr)
    sys.exit(1)

out = pathlib.Path("test-data/miracl-ko")
out.mkdir(parents=True, exist_ok=True)
(out / "qrels").mkdir(exist_ok=True)

print("Downloading MIRACL-Ko corpus (~1.5M passages, may take several minutes)...")
corpus_ds = load_dataset("miracl/miracl-corpus", "ko", split="train", trust_remote_code=True)

corpus_path = out / "corpus.jsonl"
print(f"Writing {len(corpus_ds)} passages to {corpus_path}...")
with open(corpus_path, "w") as f:
    for row in corpus_ds:
        f.write(json.dumps({
            "_id":   row["docid"],
            "title": row.get("title", ""),
            "text":  row["text"],
        }, ensure_ascii=False) + "\n")
print(f"  done ({corpus_path.stat().st_size // 1_000_000}MB)")

print("Downloading MIRACL-Ko queries + qrels...")
miracl_ds = load_dataset("miracl/miracl", "ko", trust_remote_code=True)

queries_path = out / "queries.jsonl"
qrels_path   = out / "qrels" / "test.tsv"

# Collect test split queries + qrels
seen_queries = {}
with open(queries_path, "w") as qf, open(qrels_path, "w") as rf:
    rf.write("query-id\tcorpus-id\tscore\n")
    for split in ["test"]:
        if split not in miracl_ds:
            continue
        for row in miracl_ds[split]:
            qid = row["query_id"]
            if qid not in seen_queries:
                seen_queries[qid] = True
                qf.write(json.dumps({"_id": qid, "text": row["query"]}, ensure_ascii=False) + "\n")
            for doc in row.get("positive_passages", []):
                rf.write(f"{qid}\t{doc['docid']}\t1\n")
            # MIRACL has negative_passages too but qrels only use positives (score=1)

print(f"  {len(seen_queries)} queries, qrels written")
print(f"\nDone. Dataset at test-data/miracl-ko/")
print("Index with: cargo build --features bench --release --bin eval")
print("  eval --data test-data/miracl-ko --mode all")
PYEOF
