#!/usr/bin/env bash
# router-data.sh — prepare smoltrain router bundles without touching the main research harness.

set -euo pipefail

usage() {
    cat <<'EOF'
usage:
  bash scripts/router-data.sh ko [--size N] [--pools N] [--output FILE] [--bundle-dir DIR]
  bash scripts/router-data.sh fiqa [--output FILE] [--bundle-dir DIR]
  bash scripts/router-data.sh mixed [--size N] [--pools N] [--output FILE] [--bundle-dir DIR]

notes:
  - `ko` uses MIRACL-Ko sampled signal dirs only and defaults to `--size 50000 --pools 3`.
  - `fiqa` uses only `logs/signals/fiqa`.
  - `mixed` combines FiQA and MIRACL-Ko sampled signals.
EOF
}

COMMAND="${1:-}"
if [[ -z "$COMMAND" || "$COMMAND" == "-h" || "$COMMAND" == "--help" ]]; then
    usage
    exit 0
fi
shift || true

SIZE="50000"
POOLS="3"
OUTPUT=""
BUNDLE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE="${2:?missing value for --size}"; shift 2 ;;
        --pools) POOLS="${2:?missing value for --pools}"; shift 2 ;;
        --output) OUTPUT="${2:?missing value for --output}"; shift 2 ;;
        --bundle-dir) BUNDLE_DIR="${2:?missing value for --bundle-dir}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

dirs=()
case "$COMMAND" in
    ko)
        for seed in $(seq 1 "$POOLS"); do
            dirs+=("logs/signals/miracl-ko-s${SIZE}-p${seed}")
        done
        [[ -n "$OUTPUT" ]] || OUTPUT=".bench-state/research/tier2-router-ko-s${SIZE}.jsonl"
        [[ -n "$BUNDLE_DIR" ]] || BUNDLE_DIR=".bench-state/research/tier2-router-ko-s${SIZE}-smoltrain"
        ;;
    fiqa)
        dirs=("logs/signals/fiqa")
        [[ -n "$OUTPUT" ]] || OUTPUT=".bench-state/research/tier2-router-fiqa.jsonl"
        [[ -n "$BUNDLE_DIR" ]] || BUNDLE_DIR=".bench-state/research/tier2-router-fiqa-smoltrain"
        ;;
    mixed)
        dirs=("logs/signals/fiqa")
        for seed in $(seq 1 "$POOLS"); do
            dirs+=("logs/signals/miracl-ko-s${SIZE}-p${seed}")
        done
        [[ -n "$OUTPUT" ]] || OUTPUT=".bench-state/research/tier2-router-mixed-s${SIZE}.jsonl"
        [[ -n "$BUNDLE_DIR" ]] || BUNDLE_DIR=".bench-state/research/tier2-router-mixed-s${SIZE}-smoltrain"
        ;;
    *)
        echo "unknown profile: $COMMAND" >&2
        usage
        exit 1
        ;;
esac

missing=0
for dir in "${dirs[@]}"; do
    if [[ ! -f "$dir/bm25.jsonl" || ! -f "$dir/vector.jsonl" || ! -f "$dir/tier1.jsonl" || ! -f "$dir/hybrid.jsonl" ]]; then
        echo "missing router signals in $dir — collect bm25, vector, tier1, hybrid first" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    exit 1
fi

echo "[router-data] exporting -> $OUTPUT"
python3 scripts/export-tier2-router-data.py "${dirs[@]}" --output "$OUTPUT"

echo "[router-data] preparing bundle -> $BUNDLE_DIR"
python3 scripts/prepare-tier2-router-smoltrain.py --input "$OUTPUT" --output-dir "$BUNDLE_DIR"
