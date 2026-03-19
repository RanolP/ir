#!/usr/bin/env bash
# Korean IR benchmark — Ko-StrategyQA (9,251 docs / 592 queries)
#
# Matrix:
#   Preprocessors : none (unicode61), lindera (ko)
#   Modes         : bm25, vector, hybrid, hybrid+rerank, hybrid+expand+rerank
#
# Usage:
#   scripts/bench-ko.sh              # full run
#   scripts/bench-ko.sh --reset      # delete all eval DBs, then full run
#   scripts/bench-ko.sh bm25         # only BM25 phase
#   scripts/bench-ko.sh vector       # only vector phase
#   scripts/bench-ko.sh hybrid       # only hybrid phases
#   scripts/bench-ko.sh expand       # only expand+rerank phases

set -euo pipefail

LOG="logs/bench-ko-$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
exec > >(tee "$LOG") 2>&1
echo "logging to $LOG"

DATASET="test-data/ko-strategyqa"
LINDERA="preprocessors/ko/lindera-tokenize/target/release/lindera-tokenize"

DB_RAW="test-data/ko-strategyqa-eval.sqlite"
DB_LINDERA="test-data/ko-strategyqa-eval-lindera-tokenize.sqlite"

EVAL="cargo run --release --bin eval --"
PHASE="${1:-all}"

log() { echo; echo "══ $* ══"; echo; }

assert_lindera() {
    if [[ ! -f "$LINDERA" ]]; then
        echo "error: lindera binary not built. Run:"
        echo "  cd preprocessors/ko/lindera-tokenize && cargo build --release"
        exit 1
    fi
}

# ── Phase 0: Reset ─────────────────────────────────────────────────────────────

if [[ "$PHASE" == "--reset" || "${RESET:-0}" == "1" ]]; then
    log "RESET: deleting all ko-strategyqa eval DBs"
    rm -f "$DB_RAW" "$DB_LINDERA"
    rm -f test-data/ko-strategyqa-eval*.sqlite-shm
    rm -f test-data/ko-strategyqa-eval*.sqlite-wal
    echo "  deleted"
    [[ "$PHASE" == "--reset" ]] && exit 0
fi

# ── Build ──────────────────────────────────────────────────────────────────────

log "Building eval binary"
cargo build --release --bin eval

# ── Phase 1: BM25 ─────────────────────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "bm25" ]]; then
    log "BM25 — none (unicode61)"
    $EVAL --data "$DATASET" --mode bm25

    log "BM25 — lindera (ko)"
    assert_lindera
    $EVAL --data "$DATASET" --mode bm25 --preprocessor "$LINDERA"
fi

# ── Phase 2: Vector ───────────────────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "vector" ]]; then
    log "Vector — none"
    $EVAL --data "$DATASET" --mode vector
fi

# ── Phase 3: Hybrid ───────────────────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "hybrid" ]]; then
    log "Hybrid — none"
    $EVAL --data "$DATASET" --mode hybrid

    log "Hybrid+rerank — none"
    $EVAL --data "$DATASET" --mode hybrid --rerank

    log "Hybrid+rerank — lindera (ko)"
    assert_lindera
    $EVAL --data "$DATASET" --mode hybrid --rerank --preprocessor "$LINDERA"
fi

# ── Phase 4: Expand + rerank ──────────────────────────────────────────────────
# Note: expander hurts Korean (English SFT → mixed-language sub-queries).
# Run for verification only.

if [[ "$PHASE" == "all" || "$PHASE" == "expand" ]]; then
    log "Hybrid+expand+rerank — none (Korean expansion test)"
    $EVAL --data "$DATASET" --mode hybrid --expander --rerank
fi

log "BENCHMARK COMPLETE"
echo "Results cached in:"
echo "  $DB_RAW       (raw/vector/hybrid)"
echo "  $DB_LINDERA   (lindera BM25/hybrid)"
