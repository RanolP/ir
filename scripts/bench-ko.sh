#!/usr/bin/env bash
# Korean IR benchmark — Ko-StrategyQA (9,251 docs / 592 queries)
#
# Matrix:
#   Preprocessors : none (unicode61), kiwi, mecab, lindera
#   Modes         : bm25, vector, hybrid, hybrid+rerank, hybrid+expand+rerank
#
# DB isolation: each preprocessor gets its own eval SQLite.
#   Embeddings are computed once in the base (raw) DB and stay there.
#   BM25-only runs are cheap — no model required.
#   Hybrid/rerank runs embed inside the preprocessor-specific DB (cached after first run).
#
# Usage:
#   scripts/bench-ko.sh              # full run
#   scripts/bench-ko.sh --reset      # delete all eval DBs, then full run
#   scripts/bench-ko.sh bm25         # only BM25 phase
#   scripts/bench-ko.sh vector       # only vector phase
#   scripts/bench-ko.sh hybrid       # only hybrid phases
#   scripts/bench-ko.sh expand       # only expand+rerank phases (tests expander on Korean)

set -euo pipefail

LOG="logs/bench-ko-$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
exec > >(tee "$LOG") 2>&1
echo "logging to $LOG"

DATASET="test-data/ko-strategyqa"
KIWI="preprocessors/ko/kiwi-tokenize"
MECAB="preprocessors/ko/mecab-tokenize"
LINDERA="preprocessors/ko/lindera-tokenize/target/release/lindera-tokenize"

# DB paths (derived from eval.rs naming: {corpus}-eval[-{pp_name}].sqlite)
DB_RAW="test-data/ko-strategyqa-eval.sqlite"
DB_KIWI="test-data/ko-strategyqa-eval-kiwi-tokenize.sqlite"
DB_MECAB="test-data/ko-strategyqa-eval-mecab-tokenize.sqlite"
DB_LINDERA="test-data/ko-strategyqa-eval-lindera-tokenize.sqlite"

EVAL="cargo run --release --bin eval --"
PHASE="${1:-all}"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo; echo "══ $* ══"; echo; }

assert_preprocessor() {
    local bin="$1"
    if [[ ! -x "$bin" ]]; then
        echo "error: preprocessor not found or not executable: $bin"
        exit 1
    fi
}

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
    rm -f "$DB_RAW" "$DB_KIWI" "$DB_MECAB" "$DB_LINDERA"
    rm -f test-data/ko-strategyqa-eval*.sqlite-shm
    rm -f test-data/ko-strategyqa-eval*.sqlite-wal
    echo "  deleted"
    [[ "$PHASE" == "--reset" ]] && exit 0
fi

# ── Build ──────────────────────────────────────────────────────────────────────

log "Building eval binary"
cargo build --release --bin eval

# ── Phase 1: BM25 (preprocessor comparison) ───────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "bm25" ]]; then
    log "BM25 — none (unicode61)"
    $EVAL --data "$DATASET" --mode bm25

    log "BM25 — kiwi"
    assert_preprocessor "$KIWI"
    $EVAL --data "$DATASET" --mode bm25 --preprocessor "$KIWI"

    log "BM25 — mecab"
    assert_preprocessor "$MECAB"
    $EVAL --data "$DATASET" --mode bm25 --preprocessor "$MECAB"

    log "BM25 — lindera (parity check vs mecab)"
    assert_lindera
    $EVAL --data "$DATASET" --mode bm25 --preprocessor "$LINDERA"
fi

# ── Phase 2: Vector (shared — preprocessor-independent) ───────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "vector" ]]; then
    log "Vector — none (embeddings only, ~9k docs)"
    $EVAL --data "$DATASET" --mode vector
fi

# ── Phase 3: Hybrid (score fusion, no LLM) ────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "hybrid" ]]; then
    log "Hybrid — none (base DB already has embeddings)"
    $EVAL --data "$DATASET" --mode hybrid

    log "Hybrid — kiwi"
    assert_preprocessor "$KIWI"
    $EVAL --data "$DATASET" --mode hybrid --preprocessor "$KIWI"

    log "Hybrid — mecab"
    assert_preprocessor "$MECAB"
    $EVAL --data "$DATASET" --mode hybrid --preprocessor "$MECAB"
fi

# ── Phase 4: Hybrid + rerank ───────────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "rerank" ]]; then
    log "Hybrid+rerank — none"
    $EVAL --data "$DATASET" --mode hybrid --rerank

    log "Hybrid+rerank — kiwi"
    assert_preprocessor "$KIWI"
    $EVAL --data "$DATASET" --mode hybrid --rerank --preprocessor "$KIWI"

    log "Hybrid+rerank — mecab"
    assert_preprocessor "$MECAB"
    $EVAL --data "$DATASET" --mode hybrid --rerank --preprocessor "$MECAB"
fi

# ── Phase 5: Hybrid + expand + rerank (expander on Korean) ────────────────────
# Tests whether qmd-expander-1.7B (Qwen3 base, English SFT) helps or hurts Korean.
# Hypothesis: expansion produces English sub-queries → BM25 lex hits miss; hyde/vec may help.
# Compare vs Phase 4 (same reranker, no expansion) to isolate expander effect.

if [[ "$PHASE" == "all" || "$PHASE" == "expand" ]]; then
    log "Hybrid+expand+rerank — none (Korean expansion test)"
    $EVAL --data "$DATASET" --mode hybrid --expander --rerank

    log "Hybrid+expand+rerank — kiwi"
    assert_preprocessor "$KIWI"
    $EVAL --data "$DATASET" --mode hybrid --expander --rerank --preprocessor "$KIWI"
fi

# ── Summary ────────────────────────────────────────────────────────────────────

log "BENCHMARK COMPLETE"
echo "Results cached in:"
echo "  $DB_RAW       (raw/vector/hybrid/expand)"
echo "  $DB_KIWI      (kiwi BM25/hybrid/expand)"
echo "  $DB_MECAB     (mecab BM25/hybrid)"
echo "  $DB_LINDERA   (lindera BM25 parity)"
echo
echo "To view cached results:"
echo "  sqlite3 $DB_RAW 'SELECT run_key, mode, AVG(ndcg), AVG(recall) FROM eval_run_results GROUP BY run_key, mode'"
echo
echo "For factoid/keyword retrieval benchmark (MIRACL-Korean):"
echo "  uv run scripts/download-ko-miracl.py"
echo "  cargo run --release --bin eval -- --data test-data/ko-miracl --mode bm25"
echo "  cargo run --release --bin eval -- --data test-data/ko-miracl --mode bm25 --preprocessor preprocessors/ko/kiwi-tokenize"
echo "  cargo run --release --bin eval -- --data test-data/ko-miracl --mode hybrid --rerank"
