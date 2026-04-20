#!/usr/bin/env bash
# signal-sweep.sh — Collect per-query signal data for strong-signal threshold research.
#
# Usage:
#   scripts/signal-sweep.sh [--dataset fiqa|miracl-ko|all] [--size N[,N,...]] [--pools N] [--bm25-only]
#
# Outputs to logs/signals/{dataset}[-s{size}-p{pool}]/bm25.jsonl, vector.jsonl, hybrid.jsonl
#
# Examples:
#   scripts/signal-sweep.sh --dataset fiqa
#   scripts/signal-sweep.sh --dataset miracl-ko --size 1000,5000,10000 --pools 3
#   scripts/signal-sweep.sh --dataset miracl-ko --size 50000
#
# After collection, run threshold sweep:
#   python3 scripts/threshold-sweep.py logs/signals/*/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "signal-sweep"

# ── Helpers ───────────────────────────────────────────────────────────────────

_log() { echo "[$(date +%H:%M:%S)] $*"; }

# Run a command while printing a heartbeat every 60s.
# On no-activity for 60s, also prints a STALL DETECTED warning to help diagnose hangs.
# Usage: _with_pulse <cmd> [args...]
_with_pulse() {
    "$@" &
    local child_pid=$!
    local t_start=$SECONDS
    local last_beat=$SECONDS
    while kill -0 "$child_pid" 2>/dev/null; do
        sleep 5
        local now=$SECONDS
        local since=$(( now - last_beat ))
        if [[ $since -ge 60 ]]; then
            local elapsed=$(( now - t_start ))
            _log "still running (${elapsed}s elapsed): $1"
            if [[ $elapsed -ge 120 ]]; then
                _log "STALL DETECTED — process running for ${elapsed}s with no reported completion."
                _log "  If this is 'ir update' with a Korean preprocessor, this is likely issue #13"
                _log "  (single-pipe lindera deadlock). Kill with: kill $child_pid"
                _log "  Then re-run against test-data/fixtures/miracl-ko-mini for a smaller canary."
            fi
            last_beat=$now
        fi
    done
    wait "$child_pid"
}

# ── Defaults ──────────────────────────────────────────────────────────────────

DATASET="all"
SIZES=""
POOLS=3
BM25_ONLY=0

# ── Parse args ────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --size)    SIZES="$2"; shift 2 ;;
        --pools)   POOLS="$2"; shift 2 ;;
        --bm25-only) BM25_ONLY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Build ir binary ───────────────────────────────────────────────────────────

_log "Building ir (HEAD)..."
cargo build --release --bin ir 2>&1
IR_BIN="$REPO_ROOT/target/release/ir"

# ── Helpers ───────────────────────────────────────────────────────────────────

run_signals_for() {
    local label="$1"
    local data_dir="$2"
    local collection="$3"
    local preprocessor="${4:-}"
    local out_dir="$REPO_ROOT/logs/signals/$label"

    # Skip only if a completion marker exists (written after all queries finish)
    if [[ -f "$out_dir/.done" ]]; then
        _log "[skip] $label (complete)"
        return 0
    fi

    _log "[$label] preparing collection..."
    prep_args=(
        prepare
        --ir-bin "$IR_BIN"
        --data "$data_dir"
        --collection "$collection"
    )
    [[ -n "$preprocessor" ]] && prep_args+=(--preprocessor "$preprocessor")
    [[ "$BM25_ONLY" -eq 0 ]] && prep_args+=(--embed)
    _with_pulse python3 scripts/beir-eval.py "${prep_args[@]}"

    _log "[$label] running signal collection..."
    sig_mode="all"
    [[ "$BM25_ONLY" -eq 1 ]] && sig_mode="bm25"

    python3 scripts/beir-eval.py run \
        --ir-bin "$IR_BIN" \
        --data "$data_dir" \
        --collection "$collection" \
        --mode "$sig_mode" \
        --at-k "10,20,100" \
        --signals \
        --signals-output "$out_dir"

    _log "[$label] done -> $out_dir"
}

run_sampled() {
    local base_data="$1"
    local base_label="$2"
    local preprocessor="${3:-}"
    local size="$4"

    for seed in $(seq 1 "$POOLS"); do
        local sample_label="${base_label}-s${size}-p${seed}"
        local sample_dir="$REPO_ROOT/test-data/${base_label}-s${size}-p${seed}"
        local collection="eval-${sample_label}"

        if [[ ! -f "$sample_dir/corpus.jsonl" ]]; then
            _log "Sampling: $sample_label (size=$size seed=$seed)"
            python3 scripts/beir-eval.py sample \
                --data "$base_data" \
                --size "$size" \
                --seed "$seed" \
                --output "$sample_dir"
        fi

        run_signals_for "$sample_label" "$sample_dir" "$collection" "$preprocessor"
    done
}

# ── FiQA ─────────────────────────────────────────────────────────────────────

run_fiqa() {
    local data_dir="$REPO_ROOT/test-data/fiqa"
    if [[ ! -f "$data_dir/corpus.jsonl" ]]; then
        _log "Downloading FiQA..."
        bash scripts/download-beir.sh fiqa
    fi
    run_signals_for "fiqa" "$data_dir" "eval-fiqa-signals" ""

    if [[ -n "$SIZES" ]]; then
        IFS=',' read -ra SIZE_LIST <<< "$SIZES"
        for size in "${SIZE_LIST[@]}"; do
            run_sampled "$data_dir" "fiqa" "" "$size"
        done
    fi
}

# ── MIRACL-Ko ─────────────────────────────────────────────────────────────────

run_miracl_ko() {
    local data_dir="$REPO_ROOT/test-data/miracl-ko"
    if [[ ! -f "$data_dir/corpus.jsonl" ]]; then
        _log "Downloading MIRACL-Ko (full corpus ~1.5M docs)..."
        bash scripts/download-miracl-ko.sh
    fi

    # Full corpus
    run_signals_for "miracl-ko" "$data_dir" "eval-miracl-ko-signals" "ko"

    # Sampled pools (if sizes specified)
    if [[ -n "$SIZES" ]]; then
        IFS=',' read -ra SIZE_LIST <<< "$SIZES"
        for size in "${SIZE_LIST[@]}"; do
            run_sampled "$data_dir" "miracl-ko" "ko" "$size"
        done
    fi
}

# ── Run datasets ─────────────────────────────────────────────────────────────

_log "Signal sweep: dataset=$DATASET sizes='${SIZES:-all}' pools=$POOLS bm25_only=$BM25_ONLY"

case "$DATASET" in
    fiqa)      run_fiqa ;;
    miracl-ko) run_miracl_ko ;;
    all)       run_fiqa; run_miracl_ko ;;
    *)         echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

echo ""
_log "Collection complete. Run analysis:"
_log "    python3 scripts/threshold-sweep.py logs/signals/*/"
