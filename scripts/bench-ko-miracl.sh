#!/usr/bin/env bash
# MIRACL-Korean benchmark — 213 queries, ~2.8K passages
#
# BM25 phases run in parallel (no model, isolated DBs).
# Model phases run sequentially (Metal GPU, one context).
#
# Setup:
#   uv run scripts/download-ko-miracl.py
#   cd preprocessors/ko/lindera-tokenize && cargo build --release && cd -
#
# Usage:
#   scripts/bench-ko-miracl.sh          # full run
#   scripts/bench-ko-miracl.sh bm25     # BM25 only
#   scripts/bench-ko-miracl.sh model    # model phases only

set -euo pipefail

LOG="logs/bench-ko-miracl-$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
exec > >(tee "$LOG") 2>&1
echo "logging to $LOG"

DATASET="test-data/ko-miracl"
LINDERA="preprocessors/ko/lindera-tokenize/target/release/lindera-tokenize"

EVAL="cargo run --release --bin eval --"
PHASE="${1:-all}"

log() { echo; echo "══ $* ══"; echo; }

if [[ ! -f "$DATASET/corpus.jsonl" ]]; then
    echo "error: dataset not found. Run: uv run scripts/download-ko-miracl.py"
    exit 1
fi

log "Building eval binary"
cargo build --release --bin eval

# ── BM25 (parallel) ───────────────────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "bm25" ]]; then
    log "BM25 — all preprocessors in parallel"

    $EVAL --data "$DATASET" --mode bm25 &
    PID_NONE=$!

    if [[ -x "$LINDERA" ]]; then
        $EVAL --data "$DATASET" --mode bm25 --preprocessor "$LINDERA" &
        PID_LINDERA=$!
    else
        echo "note: lindera binary not found, skipping (cd preprocessors/ko/lindera-tokenize && cargo build --release)"
        PID_LINDERA=""
    fi

    wait $PID_NONE ${PID_LINDERA:+$PID_LINDERA}
    log "BM25 complete"
fi

# ── Model phases (sequential — Metal GPU) ─────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "model" ]]; then
    log "Hybrid+rerank — none"
    $EVAL --data "$DATASET" --mode hybrid --rerank

    log "Hybrid+rerank — lindera (ko)"
    $EVAL --data "$DATASET" --mode hybrid --rerank --preprocessor "$LINDERA"

    log "Hybrid+expand+rerank — none (expander on Korean — expected to hurt)"
    $EVAL --data "$DATASET" --mode hybrid --expander --rerank
fi

log "BENCHMARK COMPLETE"
echo "Results in logs/: $LOG"
echo "To query results:"
echo "  sqlite3 test-data/ko-miracl-eval.sqlite 'SELECT run_key, mode, AVG(ndcg), AVG(recall) FROM eval_run_results GROUP BY run_key, mode'"
