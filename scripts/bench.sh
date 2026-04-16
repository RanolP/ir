#!/usr/bin/env bash
# Run IR benchmark against a BEIR dataset, optionally comparing two git revisions.
#
# Usage:
#   scripts/bench.sh <dataset> [baseline-ref] [--mode bm25|vector|hybrid|all]
#
# Examples:
#   scripts/bench.sh fiqa
#   scripts/bench.sh fiqa v0.9.0
#   scripts/bench.sh miracl-ko
#   scripts/bench.sh fiqa v0.9.0 --mode bm25
#
# Dataset names and their auto-detected settings:
#   fiqa, nfcorpus, scifact, arguana  — English BEIR datasets
#   miracl-ko                          — Korean MIRACL full corpus (download-miracl-ko.sh)
#   ko-miracl                          — Korean MIRACL dev split (download-ko-miracl.py)
#
# Results cached at logs/results/<dataset>/<git7>.json — reused on re-run.
# Comparison table printed when two versions are evaluated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Parse args ───────────────────────────────────────────────────────────────

DATASET=""
BASELINE_REF=""
MODE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --) shift; break ;;
        -*)  echo "unknown flag: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$DATASET" ]]; then
                DATASET="$1"
            elif [[ -z "$BASELINE_REF" ]]; then
                BASELINE_REF="$1"
            else
                echo "unexpected argument: $1" >&2; exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "usage: $0 <dataset> [baseline-ref] [--mode bm25|vector|hybrid|all]" >&2
    exit 1
fi

# ── Dataset config ────────────────────────────────────────────────────────────

case "$DATASET" in
    fiqa)
        DATA_PATH="test-data/fiqa"
        PREPROCESSOR=""
        DOWNLOAD_CMD="scripts/download-beir.sh fiqa"
        EMBED_FLAG="--embed"
        ;;
    nfcorpus)
        DATA_PATH="test-data/nfcorpus"
        PREPROCESSOR=""
        DOWNLOAD_CMD="scripts/download-beir.sh nfcorpus"
        EMBED_FLAG="--embed"
        ;;
    scifact)
        DATA_PATH="test-data/scifact"
        PREPROCESSOR=""
        DOWNLOAD_CMD="scripts/download-beir.sh scifact"
        EMBED_FLAG="--embed"
        ;;
    arguana)
        DATA_PATH="test-data/arguana"
        PREPROCESSOR=""
        DOWNLOAD_CMD="scripts/download-beir.sh arguana"
        EMBED_FLAG="--embed"
        ;;
    miracl-ko)
        DATA_PATH="test-data/miracl-ko"
        PREPROCESSOR="ko"
        DOWNLOAD_CMD="scripts/download-miracl-ko.sh"
        EMBED_FLAG="--embed"
        ;;
    ko-miracl)
        DATA_PATH="test-data/ko-miracl"
        PREPROCESSOR="ko"
        DOWNLOAD_CMD="uv run scripts/download-ko-miracl.py"
        EMBED_FLAG="--embed"
        ;;
    *)
        echo "unknown dataset '$DATASET'. See script header for supported datasets." >&2
        exit 1
        ;;
esac

# BM25-only mode: skip embedding
if [[ "$MODE" == "bm25" ]]; then
    EMBED_FLAG=""
fi

# ── Download dataset if missing ───────────────────────────────────────────────

if [[ ! -f "$DATA_PATH/corpus.jsonl" ]]; then
    echo "==> Dataset '$DATASET' not found. Downloading..."
    eval "$DOWNLOAD_CMD"
fi

# ── Build ir binary ───────────────────────────────────────────────────────────

echo "==> Building ir (HEAD)..."
cargo build --release --bin ir 2>&1
HEAD_BIN="$REPO_ROOT/target/release/ir"
HEAD_HASH=$(git rev-parse --short=7 HEAD)
echo "    HEAD: $HEAD_HASH"

# ── Build baseline if requested ───────────────────────────────────────────────

BASELINE_BIN=""
BASELINE_HASH=""

if [[ -n "$BASELINE_REF" ]]; then
    BASELINE_HASH=$(git rev-parse --short=7 "$BASELINE_REF" 2>/dev/null) || {
        echo "ERROR: git ref '$BASELINE_REF' not found" >&2; exit 1
    }

    if [[ "$BASELINE_HASH" == "$HEAD_HASH" ]]; then
        echo "==> Baseline $BASELINE_REF ($BASELINE_HASH) matches HEAD — reusing binary"
        BASELINE_BIN="$HEAD_BIN"
    else
        WORKTREE="/tmp/ir-bench-$BASELINE_HASH"
        if [[ ! -d "$WORKTREE" ]]; then
            echo "==> Creating worktree for $BASELINE_REF ($BASELINE_HASH)..."
            git worktree add --quiet "$WORKTREE" "$BASELINE_REF"
        fi
        echo "==> Building ir ($BASELINE_REF)..."
        cargo build --release --bin ir --manifest-path "$WORKTREE/Cargo.toml" 2>&1
        BASELINE_BIN="$WORKTREE/target/release/ir"
        echo "    baseline: $BASELINE_HASH"
    fi
fi

# ── Run one version ───────────────────────────────────────────────────────────

LOG_DIR="$REPO_ROOT/logs/results/$DATASET"
mkdir -p "$LOG_DIR"

run_version() {
    local label="$1"
    local git7="$2"
    local ir_bin="$3"
    local collection="eval-${DATASET}-${git7}"
    local result_file="$LOG_DIR/${git7}.json"

    if [[ -f "$result_file" ]]; then
        echo "==> [$label $git7] cached ($result_file)"
        return 0
    fi

    echo "==> [$label $git7] preparing collection..."
    prep_args=(
        prepare
        --ir-bin "$ir_bin"
        --data "$DATA_PATH"
        --collection "$collection"
    )
    [[ -n "$PREPROCESSOR" ]] && prep_args+=(--preprocessor "$PREPROCESSOR")
    [[ -n "$EMBED_FLAG" ]] && prep_args+=($EMBED_FLAG)
    python3 scripts/beir-eval.py "${prep_args[@]}"

    echo "==> [$label $git7] running queries (mode=$MODE)..."
    python3 scripts/beir-eval.py run \
        --ir-bin "$ir_bin" \
        --data "$DATA_PATH" \
        --collection "$collection" \
        --mode "$MODE" \
        --at-k "10,20,100" \
        --output "$result_file"

    echo "==> [$label $git7] done -> $result_file"
}

run_version "HEAD" "$HEAD_HASH" "$HEAD_BIN"
[[ -n "$BASELINE_REF" ]] && run_version "base" "$BASELINE_HASH" "$BASELINE_BIN"

# ── Print comparison table ────────────────────────────────────────────────────

print_table_row() {
    local label="$1"
    local git7="$2"
    local result_file="$LOG_DIR/${git7}.json"

    if [[ ! -f "$result_file" ]]; then
        printf "  %-22s  %-7s  %8s  %8s  %8s  %8s  %8s  %9s  %8s\n" \
               "$label" "$git7" "MISSING" "-" "-" "-" "-" "-" "-"
        return
    fi

    # Pick best mode: hybrid > vector > bm25
    local best_mode
    best_mode=$(python3 - "$result_file" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for mode in ("hybrid", "vector", "bm25"):
    if any(r["mode"] == mode for r in d["results"]):
        print(mode); break
EOF
)

    python3 - "$result_file" "$best_mode" "$label" "$git7" <<'EOF'
import json, sys
data  = json.load(open(sys.argv[1]))
bmode = sys.argv[2]
label = sys.argv[3]
git7  = sys.argv[4]

def get(results, mode, key, default="?"):
    for r in results:
        if r["mode"] == mode:
            return r["metrics"].get(key, r["timing"].get(key, default))
    return default

rs = data["results"]
n10  = get(rs, bmode, "ndcg_10")
n20  = get(rs, bmode, "ndcg_20")
r10  = get(rs, bmode, "recall_10")
r20  = get(rs, bmode, "recall_20")
r100 = get(rs, bmode, "recall_100")
r1k  = get(rs, "bm25", "recall_1000")
med  = get(rs, bmode, "median_ms")

fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
print(f"  {label:<22}  {git7:<7}  {fmt(n10):>8}  {fmt(n20):>8}  {fmt(r10):>8}  {fmt(r20):>8}  {fmt(r100):>8}  {fmt(r1k):>9}  {fmt(med):>8}")
EOF
}

echo ""
echo "══════════════════════════════════════════════════════════════════════════════════════"
echo "  Benchmark — $DATASET  (mode=$MODE)"
echo "══════════════════════════════════════════════════════════════════════════════════════"
printf "  %-22s  %-7s  %8s  %8s  %8s  %8s  %8s  %9s  %8s\n" \
       "run" "git" "nDCG@10" "nDCG@20" "R@10" "R@20" "R@100" "R@1000" "med ms"
printf "  %-22s  %-7s  %8s  %8s  %8s  %8s  %8s  %9s  %8s\n" \
       "----------------------" "-------" "--------" "--------" "--------" "--------" "--------" "---------" "--------"

print_table_row "HEAD" "$HEAD_HASH"
[[ -n "$BASELINE_REF" ]] && print_table_row "$BASELINE_REF" "$BASELINE_HASH"

echo "══════════════════════════════════════════════════════════════════════════════════════"

# ── Cleanup worktree prompt ───────────────────────────────────────────────────

if [[ -n "$BASELINE_REF" && -n "$BASELINE_HASH" && "$BASELINE_HASH" != "$HEAD_HASH" ]]; then
    WORKTREE="/tmp/ir-bench-$BASELINE_HASH"
    if [[ -d "$WORKTREE" ]]; then
        echo ""
        read -rp "Remove worktree $WORKTREE? [y/N] " ans
        if [[ "$ans" =~ ^[Yy] ]]; then
            git worktree remove --force "$WORKTREE"
            echo "Removed."
        fi
    fi
fi
