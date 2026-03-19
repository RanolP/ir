#!/usr/bin/env bash
# Compound decompounding benchmark — BM25, lindera vs none.
#
# Queries target sub-components of compound nouns: without decompounding, BM25 scores 0.
# Validates that lindera Mode::Decompose correctly indexes compound sub-parts.
#
# Setup:
#   uv run scripts/download-ko-miracl.py
#   cd preprocessors/ko/lindera-tokenize && cargo build --release && cd -
#   uv run scripts/gen-compound-bench.py
#
# Usage:
#   scripts/bench-compound.sh

set -euo pipefail

LOG="logs/bench-compound-$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
exec > >(tee "$LOG") 2>&1
echo "logging to $LOG"

DATASET="test-data/ko-compound"
LINDERA="preprocessors/ko/lindera-tokenize/target/release/lindera-tokenize"

EVAL="cargo run --release --bin eval --"

log() { echo; echo "══ $* ══"; echo; }

# Generate dataset if missing
if [[ ! -f "$DATASET/corpus.jsonl" ]]; then
    log "Generating compound benchmark dataset"
    uv run scripts/gen-compound-bench.py
fi

log "Building eval binary"
cargo build --release --bin eval

log "BM25 — none vs lindera in parallel"

$EVAL --data "$DATASET" --mode bm25 &
PID_NONE=$!

if [[ -x "$LINDERA" ]]; then
    $EVAL --data "$DATASET" --mode bm25 --preprocessor "$LINDERA" &
    PID_LINDERA=$!
else
    echo "error: lindera binary not found (cd preprocessors/ko/lindera-tokenize && cargo build --release)"
    wait $PID_NONE
    exit 1
fi

wait $PID_NONE $PID_LINDERA

log "BENCHMARK COMPLETE"
echo "Results in: $LOG"
echo "Expected: none ≈ 0.00, lindera ≫ 0.00 (compound decompounding)"
