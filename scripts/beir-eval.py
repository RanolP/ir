#!/usr/bin/env python3
"""
beir-eval.py — Drive the ir binary against a BEIR dataset and compute metrics.

Subcommands:
  prepare   Convert BEIR corpus -> ir collection (index + embed)
  run       Query collection, compute nDCG/Recall, output JSON
  sample    Sample a large corpus to a target size for threshold research
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── BEIR loading ────────────────────────────────────────────────────────────

def load_corpus(corpus_path: Path) -> dict:
    """Returns {doc_id: {title, text}}"""
    docs = {}
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docs[doc["_id"]] = {"title": doc.get("title", ""), "text": doc.get("text", "")}
    return docs


def load_queries(queries_path: Path) -> list:
    """Returns [{id, text}]"""
    queries = []
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            queries.append({"id": q["_id"], "text": q["text"]})
    return queries


def load_qrels(qrels_path: Path) -> dict:
    """Returns {query_id: {doc_id: score}}"""
    qrels = {}
    with open(qrels_path) as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if line.startswith("query-id"):
                    continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2].strip())
            if score > 0:
                qrels.setdefault(qid, {})[did] = score
    return qrels


# ── Metrics ──────────────────────────────────────────────────────────────────

def dcg(ranked: list, relevant: dict, k: int) -> float:
    total = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        rel = relevant.get(doc_id, 0)
        if rel > 0:
            total += (2 ** rel - 1) / math.log2(i + 2)
    return total


def ideal_dcg(relevant: dict, k: int) -> float:
    scores = sorted(relevant.values(), reverse=True)
    total = 0.0
    for i, rel in enumerate(scores[:k]):
        total += (2 ** rel - 1) / math.log2(i + 2)
    return total


def ndcg_at_k(ranked: list, relevant: dict, k: int) -> float:
    idcg = ideal_dcg(relevant, k)
    if idcg == 0:
        return 0.0
    return dcg(ranked, relevant, k) / idcg


def recall_at_k(ranked: list, relevant: dict, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


# ── ir CLI helpers ──────────────────────────────────────────────────────────

def run_ir(ir_bin: str, *args, check=True, capture_output=True, timeout=120, env=None) -> subprocess.CompletedProcess:
    cmd = [ir_bin] + list(args)
    return subprocess.run(cmd, capture_output=capture_output, text=True,
                          check=check, timeout=timeout, env=env)


def collection_exists(ir_bin: str, name: str) -> bool:
    try:
        result = run_ir(ir_bin, "collection", "ls")
        return name in result.stdout
    except subprocess.CalledProcessError:
        return False


def search_one(ir_bin: str, collection: str, mode: str, query: str, limit: int,
               env: dict | None = None) -> tuple:
    """Returns (ranked_doc_ids, elapsed_ms, signals). ranked_doc_ids from path field."""
    start = time.monotonic()
    try:
        result = run_ir(ir_bin, "search", "-c", collection,
                        "--mode", mode, "-n", str(limit),
                        "--json", "-q", query, timeout=60, env=env)
        elapsed_ms = (time.monotonic() - start) * 1000
        hits = json.loads(result.stdout) if result.stdout.strip() else []
        doc_ids = [h["path"].removesuffix(".txt") for h in hits]
        signals = _parse_signals(result.stderr)
        return doc_ids, elapsed_ms, signals
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        elapsed_ms = (time.monotonic() - start) * 1000
        return [], elapsed_ms, {}


def _parse_signals(stderr: str) -> dict:
    """Parse SIGNAL_BM25 and SIGNAL_FUSED lines from ir stderr."""
    signals = {}
    for line in (stderr or "").splitlines():
        if line.startswith("SIGNAL_BM25\t"):
            _, top, gap = line.split("\t")
            signals["bm25_top"] = float(top)
            signals["bm25_gap"] = float(gap)
        elif line.startswith("SIGNAL_FUSED\t"):
            _, top, gap = line.split("\t")
            signals["fused_top"] = float(top)
            signals["fused_gap"] = float(gap)
    return signals


# ── Subcommand: prepare ─────────────────────────────────────────────────────

def cmd_prepare(args):
    data_dir = Path(args.data)
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"ERROR: corpus.jsonl not found at {corpus_path}", file=sys.stderr)
        sys.exit(1)

    collection = args.collection
    corpus_dir = data_dir / "eval-corpus"
    corpus_dir.mkdir(exist_ok=True)

    # Write one .txt file per doc (skip existing files)
    print(f"Materializing corpus -> {corpus_dir}/")
    written = 0
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc["_id"]
            # Sanitize doc_id for use as filename
            safe_id = doc_id.replace("/", "_").replace("\\", "_")
            dest = corpus_dir / f"{safe_id}.txt"
            if dest.exists():
                continue
            title = doc.get("title", "")
            text = doc.get("text", "")
            content = f"{title}\n\n{text}" if title else text
            dest.write_text(content, encoding="utf-8")
            written += 1
    print(f"  {written} files written (existing skipped)")

    ir_bin = args.ir_bin

    # Create collection if it doesn't exist
    if not collection_exists(ir_bin, collection):
        add_args = ["collection", "add", collection, str(corpus_dir), "--glob", "**/*.txt"]
        if args.preprocessor:
            add_args += ["--preprocessor", args.preprocessor]
        print(f"Creating collection '{collection}'...")
        run_ir(ir_bin, *add_args, capture_output=False)

    # Index (no-op if unchanged)
    print(f"Indexing...")
    run_ir(ir_bin, "update", collection, capture_output=False, timeout=3600)

    # Embed if requested
    if args.embed:
        print(f"Embedding...")
        run_ir(ir_bin, "embed", collection, capture_output=False, timeout=7200)

    print("Done.")


# ── Subcommand: run ─────────────────────────────────────────────────────────

WARMUP = 5  # skip first N queries for latency stats


def cmd_run(args):
    data_dir = Path(args.data)
    queries = load_queries(data_dir / "queries.jsonl")
    qrels = load_qrels(data_dir / "qrels" / "test.tsv")

    # Filter to queries that have qrels
    queries = [q for q in queries if q["id"] in qrels]
    if args.max_queries:
        queries = queries[:args.max_queries]

    modes = args.mode.split(",") if "," in args.mode else (
        ["bm25", "vector", "hybrid"] if args.mode == "all" else [args.mode]
    )

    at_ks = sorted(int(k) for k in args.at_k.split(","))
    fetch_k = max(at_ks)
    fetch_k_bm25 = max(fetch_k, 1000)  # ^ always fetch 1000 for BM25 R@1000 diagnostic

    ir_bin = args.ir_bin
    collection = args.collection

    # Signal mode: run bm25+vector+hybrid with IR_BENCH_SIGNALS+IR_DISABLE_SHORTCUTS,
    # write per-query JSONL to signals_output dir.
    if args.signals:
        _run_signals(args, queries, qrels, at_ks, fetch_k, ir_bin, collection)
        return

    all_results = []

    for mode in modes:
        print(f"\n==> mode={mode} ({len(queries)} queries, k={fetch_k})")
        ranked_all = []
        latencies = []

        effective_k = fetch_k_bm25 if mode == "bm25" else fetch_k

        for i, q in enumerate(queries):
            relevant = qrels.get(q["id"], {})
            if not relevant:
                continue

            ranked, elapsed_ms, _ = search_one(ir_bin, collection, mode, q["text"], effective_k)
            ranked_all.append((q["id"], ranked, relevant))

            if i >= WARMUP:
                latencies.append(elapsed_ms)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(queries)}", end="\r", flush=True)

        print(f"  {len(ranked_all)}/{len(queries)} queries scored   ")

        if not ranked_all:
            continue

        # Aggregate metrics
        n = len(ranked_all)
        metrics = {}
        for k in at_ks:
            ndcg_sum = sum(ndcg_at_k(r, rel, k) for _, r, rel in ranked_all)
            recall_sum = sum(recall_at_k(r, rel, k) for _, r, rel in ranked_all)
            metrics[f"ndcg_{k}"] = round(ndcg_sum / n, 4)
            metrics[f"recall_{k}"] = round(recall_sum / n, 4)

        if mode == "bm25":
            recall_1000 = sum(recall_at_k(r, rel, 1000) for _, r, rel in ranked_all) / n
            metrics["recall_1000"] = round(recall_1000, 4)

        # Timing
        timing = {}
        if latencies:
            timing["median_ms"] = round(percentile(latencies, 50), 1)
            timing["p95_ms"] = round(percentile(latencies, 95), 1)

        result = {"mode": mode, "metrics": metrics, "timing": timing}
        all_results.append(result)

        # Print summary line
        ndcg_k = at_ks[0]
        print(f"  nDCG@{ndcg_k}={metrics.get(f'ndcg_{ndcg_k}', '?'):.4f}  "
              f"R@{ndcg_k}={metrics.get(f'recall_{ndcg_k}', '?'):.4f}", end="")
        if mode == "bm25":
            print(f"  R@1000={metrics.get('recall_1000', '?'):.4f}", end="")
        if timing:
            print(f"  med={timing['median_ms']}ms", end="")
        print()

    output = {
        "dataset": data_dir.name,
        "collection": collection,
        "results": all_results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        print(json.dumps(output, indent=2))


def _run_signals(args, queries, qrels, at_ks, fetch_k, ir_bin, collection):
    """Run bm25+vector+hybrid with signal capture; write per-query JSONL."""
    out_dir = Path(args.signals_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stop any running daemon so it restarts with IR_BENCH_SIGNALS in its env.
    # A daemon started without IR_BENCH_SIGNALS won't emit SIGNAL_FUSED.
    try:
        subprocess.run([ir_bin, "daemon", "stop"], capture_output=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Build subprocess env with signal instrumentation
    signal_env = os.environ.copy()
    signal_env["IR_BENCH_SIGNALS"] = "1"
    signal_env["IR_DISABLE_SHORTCUTS"] = "1"

    modes_to_run = ["bm25", "vector", "hybrid"]
    files = {m: open(out_dir / f"{m}.jsonl", "w") for m in modes_to_run}

    try:
        total = len(queries)
        for i, q in enumerate(queries):
            relevant = qrels.get(q["id"], {})
            if not relevant:
                continue

            for mode in modes_to_run:
                effective_k = max(fetch_k, 1000) if mode == "bm25" else fetch_k
                ranked, elapsed_ms, signals = search_one(
                    ir_bin, collection, mode, q["text"], effective_k, env=signal_env
                )
                rec = {
                    "query_id": q["id"],
                    "query_text": q["text"],
                    "ranked": ranked,
                    "elapsed_ms": round(elapsed_ms, 1),
                }
                # Attach signals to the mode that produces them
                rec.update(signals)
                # Per-k metrics
                for k in at_ks:
                    rec[f"ndcg{k}"] = round(ndcg_at_k(ranked, relevant, k), 6)
                    rec[f"recall{k}"] = round(recall_at_k(ranked, relevant, k), 6)
                if mode == "bm25":
                    rec["recall1000"] = round(recall_at_k(ranked, relevant, 1000), 6)
                files[mode].write(json.dumps(rec) + "\n")

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{total} queries", end="\r", flush=True)

        print(f"  {total}/{total} queries done   ")
        for mode in modes_to_run:
            print(f"  {mode} -> {out_dir}/{mode}.jsonl")
    finally:
        for f in files.values():
            f.close()


# ── Subcommand: sample ──────────────────────────────────────────────────────

def cmd_sample(args):
    data_dir = Path(args.data)
    out_dir = Path(args.output)

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"ERROR: output directory {out_dir} already exists and is non-empty", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = data_dir / "corpus.jsonl"
    qrels_dir = data_dir / "qrels"
    queries_path = data_dir / "queries.jsonl"

    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load qrels to find mandatory doc IDs
    print("Loading qrels to identify mandatory docs...")
    all_qrels = load_qrels(qrels_dir / "test.tsv")
    mandatory_ids = set()
    for doc_map in all_qrels.values():
        mandatory_ids.update(doc_map.keys())
    print(f"  {len(mandatory_ids)} mandatory docs (referenced in qrels)")

    # Stream corpus, separate mandatory from remainder
    print("Streaming corpus...")
    mandatory_docs = {}
    remainder_ids = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc["_id"]
            if doc_id in mandatory_ids:
                mandatory_docs[doc_id] = doc
            else:
                remainder_ids.append(doc_id)

    corpus_size = len(mandatory_docs) + len(remainder_ids)
    print(f"  corpus total: {corpus_size:,}  mandatory: {len(mandatory_docs):,}  remainder: {len(remainder_ids):,}")

    target = args.size
    if len(mandatory_docs) >= target:
        print(f"WARNING: mandatory docs ({len(mandatory_docs)}) >= target size ({target}). Using mandatory only.")
        sample_remainder = []
    else:
        need = target - len(mandatory_docs)
        rng = random.Random(args.seed)
        sample_remainder = rng.sample(remainder_ids, min(need, len(remainder_ids)))
        print(f"  sampling {len(sample_remainder)} remainder docs (seed={args.seed})")

    # Write sampled corpus.jsonl — need a second pass for the sampled remainder
    sampled_ids = set(mandatory_docs.keys()) | set(sample_remainder)
    print(f"Writing sampled corpus ({len(sampled_ids):,} docs) -> {out_dir}/corpus.jsonl")
    out_corpus = out_dir / "corpus.jsonl"
    written = 0
    with open(corpus_path) as f_in, open(out_corpus, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc["_id"] in sampled_ids:
                f_out.write(json.dumps(doc) + "\n")
                written += 1

    print(f"  {written} docs written")

    # Copy queries.jsonl and qrels/
    print(f"Copying queries and qrels...")
    shutil.copy2(queries_path, out_dir / "queries.jsonl")
    out_qrels = out_dir / "qrels"
    out_qrels.mkdir(exist_ok=True)
    for qrels_file in qrels_dir.iterdir():
        shutil.copy2(qrels_file, out_qrels / qrels_file.name)

    print(f"Done. Sampled dataset at {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="BEIR evaluation against a real ir collection")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared arguments
    def add_common(parser):
        parser.add_argument("--ir-bin", default="ir", help="Path to ir binary (default: ir)")
        parser.add_argument("--data", required=True, help="BEIR dataset directory")
        parser.add_argument("--collection", required=True, help="ir collection name")

    # prepare
    prep = sub.add_parser("prepare", help="Convert BEIR corpus -> ir collection")
    add_common(prep)
    prep.add_argument("--preprocessor", help="Preprocessor alias (e.g. ko for Korean)")
    prep.add_argument("--embed", action="store_true", help="Also run ir embed after indexing")

    # run
    run_p = sub.add_parser("run", help="Run queries and compute metrics")
    add_common(run_p)
    run_p.add_argument("--mode", default="bm25", help="bm25, vector, hybrid, all (default: bm25)")
    run_p.add_argument("--at-k", default="10,20,100", help="Comma-separated k values (default: 10,20,100)")
    run_p.add_argument("--max-queries", type=int, help="Limit number of queries")
    run_p.add_argument("--output", "-o", help="Write JSON results to file")
    run_p.add_argument("--signals", action="store_true",
                       help="Capture per-query signal data (bm25_top/gap, fused_top/gap). "
                            "Runs bm25+vector+hybrid with IR_BENCH_SIGNALS+IR_DISABLE_SHORTCUTS. "
                            "Requires --signals-output.")
    run_p.add_argument("--signals-output", metavar="DIR",
                       help="Directory for per-query signal JSONL files (required with --signals)")

    # sample
    samp = sub.add_parser("sample", help="Sample a large corpus to a target size")
    samp.add_argument("--data", required=True, help="Source BEIR dataset directory")
    samp.add_argument("--size", type=int, required=True, help="Target corpus size (number of docs)")
    samp.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    samp.add_argument("--output", required=True, help="Output directory for sampled dataset")

    args = p.parse_args()

    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "run":
        if args.signals and not args.signals_output:
            p.error("--signals requires --signals-output DIR")
        cmd_run(args)
    elif args.cmd == "sample":
        cmd_sample(args)


if __name__ == "__main__":
    main()
