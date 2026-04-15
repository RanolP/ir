#!/usr/bin/env bash
# Run eval sequentially across named configs, log each run, print summary.
#
# Usage:
#   scripts/bench.sh --data test-data/fiqa [--mode all] [--eval-args "..."] \
#       baseline \
#       "B:IR_COMBINED_MODEL=~/local-models/Qwen3.5-0.8B-Q8_0.gguf" \
#       "C:IR_COMBINED_MODEL=~/local-models/Qwen3.5-2B-Q4_K_M.gguf"
#
# Each run arg is either:
#   name                        — no extra env (baseline)
#   "name:KEY=VAL KEY2=VAL2"    — env vars to prepend
#
# Logs written to logs/bench-<dataset>-<name>-<timestamp>.log

set -euo pipefail

DATA=""
MODE="all"
EVAL_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)      DATA="$2";       shift 2 ;;
        --mode)      MODE="$2";       shift 2 ;;
        --eval-args) EVAL_ARGS="$2";  shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *)  break ;;
    esac
done

if [[ -z "$DATA" ]]; then
    echo "usage: $0 --data <dataset-path> [--mode all] [run-specs...]" >&2
    exit 1
fi

DATASET=$(basename "$DATA")
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Results stored as lines: "name<TAB>ndcg<TAB>recall" in a temp file.
RESULTS_FILE=$(mktemp)
trap 'rm -f "$RESULTS_FILE"' EXIT

echo "==> building eval..."
cargo build --release --bin eval --features bench 2>&1
echo ""

for spec in "$@"; do
    if [[ "$spec" == *:* ]]; then
        name="${spec%%:*}"
        env_part="${spec#*:}"
    else
        name="$spec"
        env_part=""
    fi

    log="$LOG_DIR/bench-${DATASET}-${name}-${TIMESTAMP}.log"
    json_out="$LOG_DIR/bench-${DATASET}-${name}-${TIMESTAMP}.json"
    echo "==> [$name] starting  (log: $log)"

    cmd="cargo run --release --features bench --bin eval -- --data $DATA --mode $MODE --emit-json $json_out $EVAL_ARGS"
    if [[ -n "$env_part" ]]; then
        cmd="$env_part $cmd"
    fi

    set +e
    eval "$cmd" 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    set -e

    if [[ "$status" -eq 0 ]] && [[ -f "$json_out" ]]; then
        read -r ndcg recall < <(python3 - "$json_out" <<'EOF'
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
for r in d.get("results", []):
    if r["mode"] in ("hybrid-rerank", "hybrid"):
        m = r["metrics"]
        print(m.get("ndcg_10", "?"), m.get("recall_10", "?"))
        break
else:
    print("? ?")
EOF
)
        printf '%s\t%s\t%s\n' "$name" "${ndcg:-?}" "${recall:-?}" >> "$RESULTS_FILE"
        echo ""
        echo "==> [$name] done  nDCG@10=${ndcg:-?}  Recall@10=${recall:-?}"
    else
        printf '%s\tFAILED\t-\n' "$name" >> "$RESULTS_FILE"
        echo "==> [$name] FAILED (exit $status, see $log)"
    fi
    echo ""
done

echo "══════════════════════════════════════════════════════"
echo "  Benchmark summary — $DATASET  ($TIMESTAMP)"
echo "══════════════════════════════════════════════════════"
printf "  %-20s  %10s  %12s\n" "run" "nDCG@10" "Recall@10"
printf "  %-20s  %10s  %12s\n" "--------------------" "----------" "------------"
while IFS=$'\t' read -r name ndcg recall; do
    printf "  %-20s  %10s  %12s\n" "$name" "$ndcg" "$recall"
done < "$RESULTS_FILE"
echo "══════════════════════════════════════════════════════"
