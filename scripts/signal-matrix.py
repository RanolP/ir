#!/usr/bin/env python3
"""
signal-matrix.py — Sweep multiple BM25 signal approaches × threshold range.

For each signal function, computes fire_rate and harm_rate across a threshold grid,
enabling comparison of which signal best predicts "BM25 result is good enough".

Usage:
  python3 scripts/signal-matrix.py logs/signals/fiqa/
  python3 scripts/signal-matrix.py logs/signals/fiqa/ logs/signals/miracl-ko-s5000-p1/
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


# ── Load data ────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def load_qrels(signals_dir: Path) -> dict:
    base = re.sub(r"-s\d+-p\d+$", "", signals_dir.name)
    repo_root = Path(__file__).parent.parent
    path = repo_root / "test-data" / base / "qrels" / "test.tsv"
    if not path.exists():
        print(f"WARNING: qrels not found at {path}", file=sys.stderr)
        return {}
    qrels: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels

def ndcg(ranked, rel, k=10):
    idcg = sum((2**r-1)/math.log2(i+2) for i,(d,r) in enumerate(
        sorted(rel.items(), key=lambda x:-x[1])[:k]))
    if not idcg: return 0.0
    return sum((2**rel.get(d,0)-1)/math.log2(i+2) for i,d in enumerate(ranked[:k])) / idcg


# ── Signal functions ─────────────────────────────────────────────────────────
# Each takes a query dict and returns a float signal value.
# Higher value = "BM25 result is stronger / more trustworthy".

SIGNALS = {}

def signal(name):
    def decorator(fn):
        SIGNALS[name] = fn
        return fn
    return decorator


@signal("top_gap_product")
def sig_top_gap_product(q):
    """Current tier-0: top * gap (equivalent to top >= X and gap >= Y when used as product)."""
    return q["bm25_top"] * q["bm25_gap"]

@signal("gap_only")
def sig_gap(q):
    """Raw BM25 gap (rank1 - rank2)."""
    return q["bm25_gap"]

@signal("relative_dominance")
def sig_rel_dominance(q):
    """top / mean(rank2..K) — how much rank-1 stands out relative to the field."""
    scores = q.get("bm25_scores", [])
    if len(scores) < 2:
        return 0.0
    rest = scores[1:]
    mean_rest = sum(rest) / len(rest)
    return q["bm25_top"] / mean_rest if mean_rest > 0 else 0.0

@signal("score_percentile")
def sig_percentile(q):
    """(top - min) / (max - min) within top-K result set."""
    scores = q.get("bm25_scores", [])
    if len(scores) < 2:
        return 0.0
    lo, hi = min(scores), max(scores)
    return (q["bm25_top"] - lo) / (hi - lo) if hi > lo else 0.0

@signal("tail_ratio")
def sig_tail_ratio(q):
    """top / rank_last — how far rank-1 is from the bottom of top-K."""
    scores = q.get("bm25_scores", [])
    if len(scores) < 2:
        return 0.0
    return q["bm25_top"] / scores[-1] if scores[-1] > 0 else 0.0

@signal("lexical_score")
def sig_lexical(q):
    """Query lexical density: specific terms → BM25 trustworthy."""
    text = q.get("query_text", "")
    words = text.split()
    score = 0.0
    if re.search(r'\d', text):                          score += 0.30  # has number
    if '"' in text or "'" in text:                      score += 0.25  # quoted
    if any(w[0].isupper() for w in words[1:] if w):    score += 0.20  # mid-sentence uppercase
    if len(words) <= 4:                                 score += 0.15  # short query
    if not re.match(r'^(how|why|what|when|where|who|explain|describe)', text.lower()):
                                                        score += 0.10  # no question word
    return min(score, 1.0)

@signal("lexical_x_gap")
def sig_lex_gap(q):
    """lexical_score * gap — combine query specificity with result separation."""
    return sig_lexical(q) * q["bm25_gap"]

@signal("lexical_x_relative")
def sig_lex_rel(q):
    """lexical_score * relative_dominance."""
    return sig_lexical(q) * sig_rel_dominance(q)


# ── Sweep ────────────────────────────────────────────────────────────────────

HARM_EPS = 0.01
N_THRESHOLDS = 20  # grid points per signal


def sweep_signal(signal_name, fn, queries):
    """Sweep threshold range for one signal. Returns list of (threshold, fire_rate, harm_rate)."""
    values = [fn(q) for q in queries]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return []

    thresholds = [lo + (hi - lo) * i / (N_THRESHOLDS - 1) for i in range(N_THRESHOLDS)]
    n = len(queries)
    results = []
    for t in thresholds:
        fires = [q for q, v in zip(queries, values) if v >= t]
        harmed = [q for q in fires if q["ndcg10_hybrid"] - q["ndcg10_bm25"] > HARM_EPS]
        results.append({
            "threshold": t,
            "fire_n": len(fires),
            "fire_rate": len(fires) / n,
            "harm_n": len(harmed),
            "harm_rate": len(harmed) / len(fires) if fires else 0.0,
        })
    return results


def load_dir(signals_dir: Path):
    bm25 = {r["query_id"]: r for r in load_jsonl(signals_dir / "bm25.jsonl")}
    hyb  = {r["query_id"]: r for r in load_jsonl(signals_dir / "hybrid.jsonl")}
    qrels = load_qrels(signals_dir)

    queries = []
    for qid in set(bm25) & set(hyb):
        b, h = bm25[qid], hyb[qid]
        queries.append({
            "query_id":      qid,
            "query_text":    b.get("query_text", ""),
            "bm25_top":      b.get("bm25_top", 0.0),
            "bm25_gap":      b.get("bm25_gap", 0.0),
            "bm25_scores":   b.get("bm25_scores", []),
            "ndcg10_bm25":   b.get("ndcg10", 0.0),
            "ndcg10_hybrid": h.get("ndcg10", 0.0),
        })
    return queries


# ── Report ───────────────────────────────────────────────────────────────────

def best_candidate(sweep):
    """Best threshold: harm < 5%, highest fire_rate."""
    candidates = [r for r in sweep if r["harm_rate"] < 0.05 and r["fire_rate"] > 0.01]
    return max(candidates, key=lambda r: r["fire_rate"]) if candidates else None


def print_summary(label, n, all_sweeps):
    print(f"\n{'='*72}")
    print(f"Signal Matrix — {label} ({n} queries)")
    print(f"{'='*72}")
    print(f"  {'signal':<22}  {'best_fire%':>10}  {'at_harm%':>8}  {'threshold':>10}  {'harm<5%?':>8}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    for name, sweep in all_sweeps.items():
        best = best_candidate(sweep)
        if best:
            print(f"  {name:<22}  {best['fire_rate']*100:>9.1f}%  "
                  f"{best['harm_rate']*100:>7.1f}%  {best['threshold']:>10.4f}  {'YES':>8}")
        else:
            # Best available even if harm > 5%
            safe = min(sweep, key=lambda r: r["harm_rate"]) if sweep else None
            if safe:
                print(f"  {name:<22}  {safe['fire_rate']*100:>9.1f}%  "
                      f"{safe['harm_rate']*100:>7.1f}%  {safe['threshold']:>10.4f}  {'no':>8}")
    print()


def print_detail(name, sweep):
    print(f"\n  {name}:")
    print(f"    {'threshold':>10}  {'fire%':>6}  {'harm%':>6}")
    # Print ~8 representative rows
    step = max(1, len(sweep) // 8)
    for r in sweep[::step]:
        flag = " <-- best" if r["harm_rate"] < 0.05 and r == best_candidate(sweep) else ""
        print(f"    {r['threshold']:>10.4f}  {r['fire_rate']*100:>5.1f}%  {r['harm_rate']*100:>5.1f}%{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signals_dirs", nargs="+", metavar="DIR")
    ap.add_argument("--detail", action="store_true", help="Print per-threshold rows for each signal")
    args = ap.parse_args()

    for d in args.signals_dirs:
        path = Path(d)
        print(f"\nLoading {path}...")
        queries = load_dir(path)
        print(f"  {len(queries)} queries")

        has_scores = sum(1 for q in queries if q["bm25_scores"])
        if has_scores == 0:
            print("  WARNING: no bm25_scores in data — relative_dominance/percentile/tail_ratio will be zero.")
            print("  Re-run signal collection with updated ir binary to get score arrays.")

        all_sweeps = {name: sweep_signal(name, fn, queries) for name, fn in SIGNALS.items()}
        print_summary(path.name, len(queries), all_sweeps)

        if args.detail:
            for name, sweep in all_sweeps.items():
                print_detail(name, sweep)


if __name__ == "__main__":
    main()
