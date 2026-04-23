#!/usr/bin/env bash
# research-harness.sh — Maintainer wrapper for repeatable benchmark research.
#
# Subcommands:
#   baseline   Run the recommended baseline benchmark flow
#   signals    Collect per-query signal data for threshold research
#   thresholds Collect signals (unless --analyze-only) and sweep threshold matrices
#   validate-thresholds  Shortlist threshold candidates and validate them on holdout
#
# Examples:
#   bash scripts/research-harness.sh baseline --dataset fiqa
#   bash scripts/research-harness.sh baseline --dataset miracl-ko
#   bash scripts/research-harness.sh signals --dataset miracl-ko --size 50000 --pools 3
#   bash scripts/research-harness.sh thresholds --dataset fiqa
#   bash scripts/research-harness.sh thresholds --dataset miracl-ko --size 50000 --pools 3
#   bash scripts/research-harness.sh validate-thresholds --dataset miracl-ko --size 50000 --pools 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "bench"

_log() { echo "[$(date +%H:%M:%S)] $*"; }

usage() {
    cat <<'EOF'
usage:
  bash scripts/research-harness.sh baseline   --dataset fiqa|miracl-ko [--mode bm25|vector|hybrid|all] [--size N] [--seed N]
  bash scripts/research-harness.sh signals    --dataset fiqa|miracl-ko [--size N[,N,...]] [--pools N] [--bm25-only]
  bash scripts/research-harness.sh thresholds --dataset fiqa|miracl-ko [--size N[,N,...]] [--pools N] [--bm25-only] [--analyze-only] [--output FILE]
  bash scripts/research-harness.sh validate-thresholds --dataset fiqa|miracl-ko [--size N] [--pools N] [--gate fused|bm25] [--top N] [--holdout-seed N] [--max-harm F] [--min-fire F] [--products P[,P,...]] [--output FILE]

notes:
  - `baseline --dataset miracl-ko` defaults to `--size 50000`.
  - `signals` and `thresholds` default MIRACL-Ko to sampled `--size 50000 --pools 3`.
  - Sampled threshold research uses `signal-sweep.sh --sample-only` to avoid the full 1.5M-doc corpus.
  - `validate-thresholds` reuses the locked holdout collection when the change is query-time only.
  - `--products ...` bypasses the shortlist step and validates those fused thresholds directly.
EOF
}

require_dataset() {
    if [[ -z "$DATASET" ]]; then
        echo "--dataset is required" >&2
        exit 1
    fi
}

threshold_dirs() {
    local dataset="$1"
    local sizes_csv="$2"
    local pools="$3"

    if [[ -z "$sizes_csv" ]]; then
        printf "%s\n" "logs/signals/$dataset"
        return 0
    fi

    local size
    IFS=',' read -ra sizes <<< "$sizes_csv"
    for size in "${sizes[@]}"; do
        local seed
        for seed in $(seq 1 "$pools"); do
            printf "%s\n" "logs/signals/${dataset}-s${size}-p${seed}"
        done
    done
}

default_threshold_output() {
    local dataset="$1"
    local sizes_csv="$2"
    local pools="$3"
    local label="$dataset"
    [[ -n "$sizes_csv" ]] && label="${dataset}-s${sizes_csv//,/_}-p${pools}"
    printf ".bench-state/research/%s-thresholds.json" "$label"
}

validation_run_id() {
    local head_hash
    head_hash="$(git rev-parse --short=7 HEAD)"
    if [[ -z "$(git status --short)" ]]; then
        printf "%s" "$head_hash"
        return 0
    fi

    local diff_hash
diff_hash="$(
        python3 - <<'EOF'
import hashlib
import pathlib
import subprocess

h = hashlib.sha256()
h.update(subprocess.check_output(["git", "diff", "--binary", "HEAD"]))
untracked = subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard"],
    text=True,
).splitlines()
for rel in sorted(untracked):
    p = pathlib.Path(rel)
    h.update(rel.encode("utf-8"))
    if p.is_file():
        h.update(p.read_bytes())
print(h.hexdigest()[:8], end="")
EOF
    )"
    printf "%s-wt%s" "$head_hash" "$diff_hash"
}

validation_manifest_ok() {
    local result_file="$1"
    local manifest_file="$2"
    local run_id="$3"
    local gate="$4"
    local candidate_label="$5"
    local candidate_envs="$6"
    local holdout_label="$7"
    local holdout_data="$8"
    local collection="$9"
    local baseline_file="${10}"
    local expander_model="${11}"
    local reranker_model="${12}"
    [[ -f "$result_file" && -f "$manifest_file" ]] || return 1

    python3 - "$manifest_file" "$run_id" "$gate" "$candidate_label" "$candidate_envs" \
        "$holdout_label" "$holdout_data" "$collection" "$baseline_file" \
        "$expander_model" "$reranker_model" <<'EOF'
import json
import sys

(
    manifest_file,
    run_id,
    gate,
    candidate_label,
    candidate_envs,
    holdout_label,
    holdout_data,
    collection,
    baseline_file,
    expander_model,
    reranker_model,
) = sys.argv[1:]

data = json.load(open(manifest_file))
expected = {
    "run_id": run_id,
    "gate": gate,
    "candidate_label": candidate_label,
    "candidate_envs": [s for s in candidate_envs.split(",") if s],
    "holdout_label": holdout_label,
    "holdout_data": holdout_data,
    "collection": collection,
    "baseline_file": baseline_file,
    "expander_model": expander_model,
    "reranker_model": reranker_model,
}

for key, value in expected.items():
    if data.get(key) != value:
        raise SystemExit(1)
EOF
}

write_validation_manifest() {
    local manifest_file="$1"
    local run_id="$2"
    local gate="$3"
    local candidate_label="$4"
    local candidate_envs="$5"
    local holdout_label="$6"
    local holdout_data="$7"
    local collection="$8"
    local baseline_file="$9"
    local expander_model="${10}"
    local reranker_model="${11}"

    python3 - "$manifest_file" "$run_id" "$gate" "$candidate_label" "$candidate_envs" \
        "$holdout_label" "$holdout_data" "$collection" "$baseline_file" \
        "$expander_model" "$reranker_model" <<'EOF'
import json
import sys
from pathlib import Path

(
    manifest_file,
    run_id,
    gate,
    candidate_label,
    candidate_envs,
    holdout_label,
    holdout_data,
    collection,
    baseline_file,
    expander_model,
    reranker_model,
) = sys.argv[1:]

Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
with open(manifest_file, "w") as f:
    json.dump(
        {
            "run_id": run_id,
            "gate": gate,
            "candidate_label": candidate_label,
            "candidate_envs": [s for s in candidate_envs.split(",") if s],
            "holdout_label": holdout_label,
            "holdout_data": holdout_data,
            "collection": collection,
            "baseline_file": baseline_file,
            "expander_model": expander_model,
            "reranker_model": reranker_model,
        },
        f,
        indent=2,
    )
EOF
}

collection_ready_for_mode() {
    local collection="$1"
    local mode="$2"
    local db_path="$IR_CONFIG_DIR/collections/${collection}.sqlite"
    [[ -f "$db_path" ]] || return 1

    python3 - "$db_path" "$mode" <<'EOF'
import sqlite3
import sys

db_path, requested_mode = sys.argv[1:]
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    active = cur.execute(
        "SELECT COUNT(*) FROM documents WHERE active = 1"
    ).fetchone()[0]
    if active <= 0:
        raise SystemExit(1)
    if requested_mode == "bm25":
        raise SystemExit(0)
    pending = cur.execute(
        """
        SELECT COUNT(*)
        FROM documents d
        WHERE d.active = 1
          AND NOT EXISTS (
              SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash
          )
        """
    ).fetchone()[0]
    raise SystemExit(0 if pending == 0 else 1)
except sqlite3.Error:
    raise SystemExit(1)
EOF
}

run_baseline() {
    require_dataset
    if [[ "$DATASET" == "miracl-ko" && -z "$SIZES" ]]; then
        SIZES="50000"
    fi

    _log "Pre-ship gate (bm25-only)"
    bash scripts/preship.sh --bm25-only

    local args=(bash scripts/bench.sh "$DATASET")
    [[ -n "$MODE" ]] && args+=(--mode "$MODE")
    if [[ -n "$SIZES" ]]; then
        if [[ "$SIZES" == *,* ]]; then
            echo "baseline supports a single --size value" >&2
            exit 1
        fi
        args+=(--size "$SIZES" --seed "$SEED")
    fi
    "${args[@]}"
}

run_signals() {
    require_dataset
    local args=(bash scripts/signal-sweep.sh --dataset "$DATASET")

    if [[ "$DATASET" == "miracl-ko" && -z "$SIZES" ]]; then
        SIZES="50000"
    fi
    if [[ -n "$SIZES" ]]; then
        args+=(--size "$SIZES" --sample-only)
    fi
    [[ -n "$POOLS" ]] && args+=(--pools "$POOLS")
    [[ "$BM25_ONLY" -eq 1 ]] && args+=(--bm25-only)

    _log "Pre-ship gate (bm25-only)"
    bash scripts/preship.sh --bm25-only
    "${args[@]}"
}

run_thresholds() {
    require_dataset
    if [[ "$DATASET" == "miracl-ko" && -z "$SIZES" ]]; then
        SIZES="50000"
    fi

    if [[ "$ANALYZE_ONLY" -eq 0 ]]; then
        run_signals
    fi

    local dirs=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && dirs+=("$line")
    done < <(threshold_dirs "$DATASET" "$SIZES" "$POOLS")

    if [[ ${#dirs[@]} -eq 0 ]]; then
        echo "no signal directories selected" >&2
        exit 1
    fi

    mkdir -p .bench-state/research
    local output="$OUTPUT"
    if [[ -z "$output" ]]; then
        output="$(default_threshold_output "$DATASET" "$SIZES" "$POOLS")"
    fi

    _log "Analyzing threshold matrices -> $output"
    python3 scripts/threshold-sweep.py "${dirs[@]}" --output "$output"
}

run_validate_thresholds() {
    require_dataset
    if [[ "$DATASET" == "miracl-ko" && -z "$SIZES" ]]; then
        SIZES="50000"
    fi
    if [[ -n "$SIZES" && "$SIZES" == *,* ]]; then
        echo "validate-thresholds supports a single --size value" >&2
        exit 1
    fi

    mkdir -p .bench-state/research
    local sweep_output="$OUTPUT"
    if [[ -z "$sweep_output" ]]; then
        sweep_output="$(default_threshold_output "$DATASET" "$SIZES" "$POOLS")"
    fi
    if [[ ! -f "$sweep_output" ]]; then
        _log "Threshold sweep missing -> $sweep_output"
        local saved_output="$OUTPUT"
        OUTPUT="$sweep_output"
        run_thresholds
        OUTPUT="$saved_output"
    fi

    local size_label=""
    [[ -n "$SIZES" ]] && size_label="-s${SIZES}"
    local shortlist=".bench-state/research/${DATASET}${size_label}-${GATE}-candidates.json"
    if [[ -n "$PRODUCTS" ]]; then
        if [[ "$GATE" != "fused" ]]; then
            echo "--products is only supported with --gate fused" >&2
            exit 1
        fi
        _log "Using explicit fused products -> $shortlist"
        python3 - "$PRODUCTS" "$shortlist" <<'EOF'
import json
import sys
from pathlib import Path

products_csv, output = sys.argv[1:]
rows = []
for raw in products_csv.split(","):
    raw = raw.strip()
    if not raw:
        continue
    rows.append({
        "gate": "fused",
        "product": float(raw),
        "label": f"product-{raw.replace('-', 'm').replace('.', '_')}",
        "passes": True,
        "source": "explicit",
    })

if not rows:
    raise SystemExit("no valid products provided")

Path(output).parent.mkdir(parents=True, exist_ok=True)
with open(output, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Explicit shortlist written to {output}")
EOF
    else
        _log "Shortlisting $GATE candidates -> $shortlist"
        python3 scripts/threshold-validate.py "$sweep_output" \
            --gate "$GATE" \
            --top "$TOP_N" \
            --max-harm "$MAX_HARM" \
            --min-fire "$MIN_FIRE" \
            --output "$shortlist"
    fi

    local candidate_count
    candidate_count="$(python3 - "$shortlist" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
print(len(rows))
EOF
)"
    if [[ "$candidate_count" == "0" ]]; then
        echo "no passing candidates in $shortlist" >&2
        exit 1
    fi

    _log "Building ir (HEAD)..."
    cargo build --release --bin ir
    local ir_bin="$REPO_ROOT/target/release/ir"
    local head_hash
    head_hash="$(git rev-parse --short=7 HEAD)"
    local run_id
    run_id="$(validation_run_id)"

    local holdout_label=""
    local holdout_data=""
    local preprocessor=""
    unset IR_COMBINED_MODEL IR_QWEN_MODEL
    export IR_EXPANDER_MODEL="${IR_EXPANDER_MODEL:-tobil/qmd-query-expansion-1.7B}"
    export IR_RERANKER_MODEL="${IR_RERANKER_MODEL:-ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF}"
    case "$DATASET" in
        fiqa)
            holdout_label="fiqa"
            holdout_data="test-data/fiqa"
            ;;
        miracl-ko)
            holdout_label="miracl-ko-s${SIZES}-p${HOLDOUT_SEED}"
            holdout_data="test-data/${holdout_label}"
            preprocessor="ko"
            if [[ ! -f "$holdout_data/corpus.jsonl" ]]; then
                _log "Sampling MIRACL-Ko holdout -> ${holdout_label}"
                python3 scripts/beir-eval.py sample \
                    --data "test-data/miracl-ko" \
                    --size "$SIZES" \
                    --seed "$HOLDOUT_SEED" \
                    --output "$holdout_data"
            fi
            ;;
        *)
            echo "validate-thresholds is not configured for dataset '$DATASET'" >&2
            exit 1
            ;;
    esac

    local baseline_file="logs/results/${holdout_label}/${head_hash}.json"
    if [[ ! -f "$baseline_file" ]]; then
        echo "missing baseline $baseline_file — run baseline lock first" >&2
        exit 1
    fi

    local collection="eval-${holdout_label}-${head_hash}"
    if collection_ready_for_mode "$collection" "hybrid"; then
        _log "Holdout collection ready -> $collection"
    else
        _log "Preparing holdout collection -> $collection"
        local prep_args=(
            prepare
            --ir-bin "$ir_bin"
            --data "$holdout_data"
            --collection "$collection"
            --embed
        )
        [[ -n "$preprocessor" ]] && prep_args+=(--preprocessor "$preprocessor")
        bench_run_guarded "prepare ${collection}" "$ir_bin" python3 scripts/beir-eval.py "${prep_args[@]}"
    fi

    local validate_dir=".bench-state/research/validate/${holdout_label}"
    mkdir -p "$validate_dir"

    while IFS=$'\t' read -r candidate_label candidate_envs; do
        [[ -n "$candidate_label" ]] || continue
        local result_file="${validate_dir}/${run_id}-${GATE}-${candidate_label}.json"
        local manifest_file="${validate_dir}/${run_id}-${GATE}-${candidate_label}.meta.json"
        if validation_manifest_ok \
            "$result_file" \
            "$manifest_file" \
            "$run_id" \
            "$GATE" \
            "$candidate_label" \
            "$candidate_envs" \
            "$holdout_label" \
            "$holdout_data" \
            "$collection" \
            "$baseline_file" \
            "$IR_EXPANDER_MODEL" \
            "$IR_RERANKER_MODEL"
        then
            _log "[$candidate_label] cached -> $result_file"
            continue
        fi

        _log "[$candidate_label] validating on ${holdout_label}"
        "$ir_bin" daemon stop >/dev/null 2>&1 || true

        local env_args=()
        while IFS= read -r env_pair; do
            [[ -n "$env_pair" ]] || continue
            env_args+=("$env_pair")
        done < <(printf '%s\n' "$candidate_envs" | tr ',' '\n')

        bench_run_guarded "validate ${collection} ${candidate_label}" "$ir_bin" env "${env_args[@]}" \
            python3 scripts/beir-eval.py run \
            --ir-bin "$ir_bin" \
            --data "$holdout_data" \
            --collection "$collection" \
            --mode hybrid \
            --at-k "10,20,100" \
            --output "$result_file"

        write_validation_manifest \
            "$manifest_file" \
            "$run_id" \
            "$GATE" \
            "$candidate_label" \
            "$candidate_envs" \
            "$holdout_label" \
            "$holdout_data" \
            "$collection" \
            "$baseline_file" \
            "$IR_EXPANDER_MODEL" \
            "$IR_RERANKER_MODEL"
    done < <(python3 - "$shortlist" "$GATE" "$preprocessor" <<'EOF'
import json
import sys

rows = json.load(open(sys.argv[1]))
gate = sys.argv[2]
preprocessor = sys.argv[3]

for row in rows:
    label = row.get("label")
    if gate == "fused":
        product = float(row["product"])
        product_str = format(product, ".15g")
        if not label:
            label = f"product-{product_str.replace('-', 'm').replace('.', '_')}"
        key = (
            f"IR_STRONG_SIGNAL_PRODUCT_PREPROCESSED_OVERRIDE={product_str}"
            if preprocessor
            else f"IR_STRONG_SIGNAL_PRODUCT_OVERRIDE={product_str}"
        )
        print(f"{label}\t{key}")
    else:
        floor = float(row["floor"])
        gap = float(row["gap"])
        floor_str = format(floor, ".15g")
        gap_str = format(gap, ".15g")
        if not label:
            label = (
                f"floor-{floor_str.replace('-', 'm').replace('.', '_')}"
                f"-gap-{gap_str.replace('-', 'm').replace('.', '_')}"
            )
        print(
            f"{label}\t"
            f"IR_BM25_STRONG_FLOOR_OVERRIDE={floor_str},"
            f"IR_BM25_STRONG_GAP_OVERRIDE={gap_str}"
        )
EOF
)

    python3 - "$baseline_file" "$validate_dir" "$run_id" "$GATE" "$shortlist" <<'EOF'
import json
import sys
from pathlib import Path

baseline_file, validate_dir, run_id, gate, shortlist = sys.argv[1:]
baseline = json.load(open(baseline_file))
baseline_hybrid = next(r for r in baseline["results"] if r["mode"] == "hybrid")
base_metrics = baseline_hybrid["metrics"]
base_timing = baseline_hybrid.get("timing", {})
rows = json.load(open(shortlist))

print("")
print("Validation — holdout hybrid")
print("=" * 88)
print(f"  {'candidate':<24}  {'nDCG@10':>8}  {'ΔnDCG':>8}  {'R@10':>8}  {'ΔR@10':>8}  {'med ms':>8}  {'Δmed%':>8}")
print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
print(
    f"  {'baseline':<24}  {float(base_metrics['ndcg_10']):>8.4f}  {0.0:>8.4f}  "
    f"{float(base_metrics['recall_10']):>8.4f}  {0.0:>8.4f}  {float(base_timing['median_ms']):>8.1f}  {0.0:>7.1f}%"
)
default_candidate = None
for row in rows:
    label = row.get("label")
    if not label:
        if gate == "fused":
            label = f"product-{format(float(row['product']), '.15g').replace('-', 'm').replace('.', '_')}"
        else:
            label = (
                f"floor-{format(float(row['floor']), '.15g').replace('-', 'm').replace('.', '_')}"
                f"-gap-{format(float(row['gap']), '.15g').replace('-', 'm').replace('.', '_')}"
            )
    result_file = Path(validate_dir) / f"{run_id}-{gate}-{label}.json"
    if not result_file.exists():
        continue
    data = json.load(open(result_file))
    hybrid = next(r for r in data["results"] if r["mode"] == "hybrid")
    metrics = hybrid["metrics"]
    timing = hybrid.get("timing", {})
    n10 = float(metrics["ndcg_10"])
    r10 = float(metrics["recall_10"])
    med = float(timing["median_ms"])
    delta_n10 = n10 - float(base_metrics["ndcg_10"])
    delta_r10 = r10 - float(base_metrics["recall_10"])
    delta_med = ((med / float(base_timing["median_ms"])) - 1.0) * 100.0
    if gate == "fused" and abs(float(row.get("product", -1.0)) - 0.06) < 1e-9:
        default_candidate = (label, delta_n10, delta_r10, delta_med)
    elif gate == "bm25" and abs(float(row.get("floor", -1.0)) - 0.75) < 1e-9 and abs(float(row.get("gap", -1.0)) - 0.10) < 1e-9:
        default_candidate = (label, delta_n10, delta_r10, delta_med)
    print(
        f"  {label:<24}  {n10:>8.4f}  {delta_n10:>8.4f}  "
        f"{r10:>8.4f}  {delta_r10:>8.4f}  {med:>8.1f}  {delta_med:>7.1f}%"
    )

if default_candidate is not None:
    label, delta_n10, delta_r10, delta_med = default_candidate
    if abs(delta_n10) > 0.005 or abs(delta_r10) > 0.005:
        print(
            f"\nSANITY FAIL: default-threshold candidate {label} diverged from baseline "
            f"(ΔnDCG={delta_n10:.4f}, ΔR@10={delta_r10:.4f}, Δmed%={delta_med:.1f}%).",
            file=sys.stderr,
        )
        print(
            "This indicates a validator/pipeline mismatch or non-threshold behavior change in the working tree.",
            file=sys.stderr,
        )
        raise SystemExit(2)
EOF
}

COMMAND="${1:-}"
if [[ -z "$COMMAND" || "$COMMAND" == "-h" || "$COMMAND" == "--help" ]]; then
    usage
    exit 0
fi
shift || true

DATASET=""
MODE="all"
SIZES=""
SEED="42"
POOLS="3"
BM25_ONLY=0
ANALYZE_ONLY=0
OUTPUT=""
GATE="fused"
TOP_N="3"
HOLDOUT_SEED="42"
MAX_HARM="0.05"
MIN_FIRE="0.10"
PRODUCTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --size) SIZES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --pools) POOLS="$2"; shift 2 ;;
        --bm25-only) BM25_ONLY=1; shift ;;
        --analyze-only) ANALYZE_ONLY=1; shift ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --gate) GATE="$2"; shift 2 ;;
        --top) TOP_N="$2"; shift 2 ;;
        --holdout-seed) HOLDOUT_SEED="$2"; shift 2 ;;
        --max-harm) MAX_HARM="$2"; shift 2 ;;
        --min-fire) MIN_FIRE="$2"; shift 2 ;;
        --products) PRODUCTS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

case "$COMMAND" in
    baseline) run_baseline ;;
    signals) run_signals ;;
    thresholds) run_thresholds ;;
    validate-thresholds) run_validate_thresholds ;;
    *) echo "unknown subcommand: $COMMAND" >&2; usage; exit 1 ;;
esac
