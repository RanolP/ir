#!/usr/bin/env bash
# pool-size-study.sh — One-off variance study for pool sizing.
#
# Runs miracl-ko at sizes [500, 1000, 2000, 5000, 10000] × N seeds (default 5)
# in BM25-only mode to keep wall-clock under ~30 minutes.
# Outputs per-(size,seed) signal JSONL to logs/signals/, then calls
# scripts/pool-size-aggregate.py to produce research/pool-size-study.md.
#
# Run once when: first setting up the bench harness, or after a major pipeline change.
#
# Usage:
#   scripts/pool-size-study.sh
#   scripts/pool-size-study.sh --seeds 3 --sizes 500,1000,2000
#   scripts/pool-size-study.sh --resume     # skip (size,seed) pairs that already have .done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "pool-size-study"

_log() { echo "[$(date +%H:%M:%S)] $*"; }

SIZES="500,1000,2000,5000,10000"
SEEDS=5
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes) SIZES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SOURCE_DATA="$REPO_ROOT/test-data/miracl-ko"
if [[ ! -f "$SOURCE_DATA/corpus.jsonl" ]]; then
    echo "ERROR: MIRACL-Ko corpus not found at $SOURCE_DATA/corpus.jsonl" >&2
    echo "       Run: bash scripts/download-miracl-ko.sh" >&2
    exit 1
fi

MANDATORY_DOCS=$(python3 - <<'PY'
ids = set()
with open("test-data/miracl-ko/qrels/test.tsv") as f:
    next(f)
    for line in f:
        _qid, docid, _score = line.rstrip("\n").split("\t")
        ids.add(docid)
print(len(ids))
PY
)

# Build ir binary
_log "Building ir (HEAD)..."
cargo build --release --bin ir 2>&1
IR_BIN="$REPO_ROOT/target/release/ir"

IFS=',' read -ra SIZE_LIST <<< "$SIZES"

_log "Pool-size study: sizes=${SIZES} seeds=${SEEDS} (bm25-only)"
_log "Total runs: $(( ${#SIZE_LIST[@]} * SEEDS ))"
_log "Mandatory qrel-linked docs: ${MANDATORY_DOCS}"

for size in "${SIZE_LIST[@]}"; do
    if [[ "$size" -le "$MANDATORY_DOCS" ]]; then
        _log "[skip] size=$size <= mandatory_docs=$MANDATORY_DOCS (deterministic sample, no between-seed variance)"
        continue
    fi
    for seed in $(seq 1 "$SEEDS"); do
        label="miracl-ko-s${size}-p${seed}"
        sample_dir="$REPO_ROOT/test-data/${label}"
        out_dir="$REPO_ROOT/logs/signals/${label}"
        collection="eval-${label}"

        if [[ "$RESUME" -eq 1 && -f "$out_dir/.done" ]]; then
            _log "[skip] $label (complete)"
            continue
        fi

        # Sample corpus if not already done
        if [[ ! -f "$sample_dir/corpus.jsonl" ]]; then
            _log "Sampling: size=$size seed=$seed -> $sample_dir"
            python3 scripts/beir-eval.py sample \
                --data "$SOURCE_DATA" \
                --size "$size" \
                --seed "$seed" \
                --output "$sample_dir"
        fi

        # Prepare collection (BM25 only — no embed)
        _log "Preparing: $label"
        python3 scripts/beir-eval.py prepare \
            --ir-bin "$IR_BIN" \
            --data "$sample_dir" \
            --collection "$collection" \
            --preprocessor ko

        # Run signal collection (bm25 only)
        _log "Collecting signals: $label"
        python3 scripts/beir-eval.py run \
            --ir-bin "$IR_BIN" \
            --data "$sample_dir" \
            --collection "$collection" \
            --mode bm25 \
            --at-k "10,20,100" \
            --signals \
            --signals-output "$out_dir"

        _log "Done: $label -> $out_dir"
    done
done

echo ""
_log "All runs complete. Aggregating results..."
python3 scripts/pool-size-aggregate.py \
    --sizes "$SIZES" \
    --seeds "$SEEDS" \
    --signals-root "$REPO_ROOT/logs/signals" \
    --output "$REPO_ROOT/research/pool-size-study.md"

_log "Study written to research/pool-size-study.md"
