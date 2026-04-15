#!/usr/bin/env bash
# compare.sh — Compare eval scores across git versions and configs.
#
# Usage:
#   scripts/compare.sh --data test-data/fiqa [--against <git-ref>] \
#       [--mode all] [--eval-args "..."] \
#       baseline \
#       "B:IR_COMBINED_MODEL=~/local-models/Qwen3.5-0.8B-Q8_0.gguf"
#
# Each run spec: "name" or "name:KEY=VAL KEY2=VAL2"
# --against builds that git ref in a worktree and runs it as "against/<ref>".
# Scores are cached in logs/results/{dataset}/{git7}-{key_hash}.json.
# If a JSON already exists for a (version, config) pair, it is reused.
#
# Requires: cargo, git, jq
set -euo pipefail

DATA=""
MODE="all"
EVAL_ARGS=""
AGAINST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)       DATA="$2";       shift 2 ;;
        --mode)       MODE="$2";       shift 2 ;;
        --eval-args)  EVAL_ARGS="$2";  shift 2 ;;
        --against)    AGAINST="$2";    shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

if [[ -z "$DATA" ]]; then
    echo "usage: $0 --data <dataset-path> [--against <git-ref>] [run-specs...]" >&2
    exit 1
fi

DATASET=$(basename "$DATA")
RESULTS_DIR="logs/results/$DATASET"
mkdir -p "$RESULTS_DIR" logs/

# ── Helpers ───────────────────────────────────────────────────────────────────

run_key_hash() {
    # hash the env+args string to 8 hex chars
    local spec="$1"
    printf '%s' "$spec" | shasum -a 256 | cut -c1-8
}

result_path() {
    local git7="$1" key_hash="$2"
    echo "$RESULTS_DIR/${git7}-${key_hash}.json"
}

build_eval() {
    local src_dir="$1"
    echo "==> building eval in $src_dir..."
    cargo build --release --bin eval --features bench --manifest-path "$src_dir/Cargo.toml" 2>&1
    echo ""
}

run_spec_to_parts() {
    # split "name:KEY=VAL" into name and env_part
    local spec="$1"
    if [[ "$spec" == *:* ]]; then
        echo "${spec%%:*}" "${spec#*:}"
    else
        echo "$spec" ""
    fi
}

collect_metric() {
    # extract a single metric from JSON result file
    local file="$1" mode="$2" metric="$3"
    jq -r ".results[] | select(.mode == \"$mode\") | .metrics.$metric // \"—\"" "$file" 2>/dev/null || echo "—"
}

collect_timing() {
    local file="$1" mode="$2"
    jq -r ".results[] | select(.mode == \"$mode\") | .timing.median_ms // \"—\"" "$file" 2>/dev/null || echo "—"
}

# ── Build current HEAD ────────────────────────────────────────────────────────

CURRENT_HASH=$(git rev-parse --short=7 HEAD)
echo "==> current HEAD: $CURRENT_HASH"
build_eval "."

CURRENT_EVAL="./target/release/eval"

# ── Optionally build --against ref ───────────────────────────────────────────

AGAINST_HASH=""
AGAINST_EVAL=""
AGAINST_WORKTREE=""

if [[ -n "$AGAINST" ]]; then
    AGAINST_HASH=$(git rev-parse --short=7 "$AGAINST")
    echo "==> against: $AGAINST ($AGAINST_HASH)"

    if [[ "$AGAINST_HASH" == "$CURRENT_HASH" ]]; then
        echo "  same commit as HEAD, reusing current binary"
        AGAINST_EVAL="$CURRENT_EVAL"
    else
        AGAINST_WORKTREE="/tmp/ir-eval-$AGAINST_HASH"
        if [[ -d "$AGAINST_WORKTREE" ]]; then
            echo "  worktree already exists at $AGAINST_WORKTREE"
        else
            git worktree add "$AGAINST_WORKTREE" "$AGAINST" 2>&1
        fi
        build_eval "$AGAINST_WORKTREE"
        AGAINST_EVAL="$AGAINST_WORKTREE/target/release/eval"
    fi
fi

# ── Run variants ─────────────────────────────────────────────────────────────

# Collect: (label, git7, eval_binary, env_part) tuples stored in parallel arrays
LABELS=()
GIT7S=()
BINARIES=()
ENV_PARTS=()
JSON_PATHS=()

add_variant() {
    local label="$1" git7="$2" binary="$3" env_part="$4"
    local run_desc="mode=$MODE|args=$EVAL_ARGS|env=$env_part"
    local key_hash
    key_hash=$(run_key_hash "$run_desc")
    local json_path
    json_path=$(result_path "$git7" "$key_hash")
    LABELS+=("$label")
    GIT7S+=("$git7")
    BINARIES+=("$binary")
    ENV_PARTS+=("$env_part")
    JSON_PATHS+=("$json_path")
}

for spec in "$@"; do
    read -r name env_part <<< "$(run_spec_to_parts "$spec")"
    add_variant "$name" "$CURRENT_HASH" "$CURRENT_EVAL" "$env_part"
    if [[ -n "$AGAINST" ]] && [[ -n "$AGAINST_HASH" ]]; then
        add_variant "against/$name" "$AGAINST_HASH" "$AGAINST_EVAL" "$env_part"
    fi
done

if [[ ${#LABELS[@]} -eq 0 ]]; then
    # default: just run baseline for current and against
    add_variant "baseline" "$CURRENT_HASH" "$CURRENT_EVAL" ""
    if [[ -n "$AGAINST" ]] && [[ -n "$AGAINST_HASH" ]]; then
        add_variant "against/baseline" "$AGAINST_HASH" "$AGAINST_EVAL" ""
    fi
fi

# ── Execute missing runs ──────────────────────────────────────────────────────

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    git7="${GIT7S[$i]}"
    binary="${BINARIES[$i]}"
    env_part="${ENV_PARTS[$i]}"
    json_path="${JSON_PATHS[$i]}"

    if [[ -f "$json_path" ]]; then
        echo "==> [$label] cached ($json_path)"
        continue
    fi

    echo "==> [$label] running ($git7)..."
    log="logs/compare-$DATASET-${label//\//-}-$(date +%Y%m%d-%H%M%S).log"

    cmd="$binary --data $DATA --mode $MODE --emit-json $json_path --version-tag $git7 $EVAL_ARGS"
    if [[ -n "$env_part" ]]; then
        cmd="$env_part $cmd"
    fi

    set +e
    eval "$cmd" 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    set -e

    if [[ "$status" -ne 0 ]]; then
        echo "==> [$label] FAILED (exit $status, see $log)"
        JSON_PATHS[$i]=""
    else
        echo "==> [$label] done → $json_path"
    fi
    echo ""
done

# ── Print comparison table ────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  $DATASET comparison"
echo "══════════════════════════════════════════════════════════════════════"
printf "  %-22s  %-6s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %s\n" \
    "run" "git" "nDCG@10" "nDCG@20" "R@10" "R@20" "R@100" "R@1000" "med ms"
printf "  %-22s  %-6s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %s\n" \
    "----------------------" "------" "--------" "--------" "--------" "--------" "--------" "--------" "------"

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    git7="${GIT7S[$i]}"
    json_path="${JSON_PATHS[$i]}"

    if [[ -z "$json_path" ]] || [[ ! -f "$json_path" ]]; then
        printf "  %-22s  %-6s  FAILED\n" "$label" "$git7"
        continue
    fi

    # find the best mode (hybrid-rerank > hybrid > vector > bm25)
    best_mode=$(jq -r '
        .results | map(.mode) |
        if index("hybrid-rerank") then "hybrid-rerank"
        elif index("hybrid") then "hybrid"
        elif index("vector") then "vector"
        else "bm25" end' "$json_path" 2>/dev/null || echo "bm25")

    n10=$(collect_metric "$json_path" "$best_mode" "ndcg_10")
    n20=$(collect_metric "$json_path" "$best_mode" "ndcg_20")
    r10=$(collect_metric "$json_path" "$best_mode" "recall_10")
    r20=$(collect_metric "$json_path" "$best_mode" "recall_20")
    r100=$(collect_metric "$json_path" "$best_mode" "recall_100")
    r1000=$(collect_metric "$json_path" "bm25" "recall_1000")
    ms=$(collect_timing "$json_path" "$best_mode")

    printf "  %-22s  %-6s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %s\n" \
        "$label" "$git7" "$n10" "$n20" "$r10" "$r20" "$r100" "$r1000" "$ms"
done

echo "══════════════════════════════════════════════════════════════════════"

# ── Cleanup worktree ─────────────────────────────────────────────────────────

if [[ -n "$AGAINST_WORKTREE" ]] && [[ -d "$AGAINST_WORKTREE" ]]; then
    echo ""
    read -rp "Remove worktree $AGAINST_WORKTREE? [y/N] " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        git worktree remove "$AGAINST_WORKTREE" --force
        echo "removed"
    fi
fi
