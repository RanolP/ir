#!/usr/bin/env bash
# preship.sh — Pre-ship regression gate for ir.
#
# Runs committed fixtures through three axes:
#   stability — no hang, no crash, within wall-clock budget
#   speed     — docs/sec >= minimum, query p50 <= maximum
#   performance — nDCG@10 and Recall@10 within tolerance of expected values
#
# Uncalibrated fixtures (calibrated=false in expected.json) run perf in WARN mode.
# Calibrate with: scripts/calibrate-fixtures.sh <fixture-name>
#
# Exit codes: 0=all PASS, 1=any FAIL, 2=WARN-only (perf drift, no hard failure)
#
# Usage:
#   scripts/preship.sh
#   scripts/preship.sh --bm25-only          # skip embed, BM25 assertions only
#   scripts/preship.sh --fixture synthetic-en
#   scripts/preship.sh --only stability

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "preship"

BM25_ONLY=0
FIXTURE_FILTER=""
RUN_STABILITY=1
RUN_SPEED=1
RUN_PERF=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bm25-only)    BM25_ONLY=1; shift ;;
        --fixture)      FIXTURE_FILTER="$2"; shift 2 ;;
        --only)
            RUN_STABILITY=0; RUN_SPEED=0; RUN_PERF=0
            case "$2" in
                stability) RUN_STABILITY=1 ;;
                speed)     RUN_SPEED=1 ;;
                perf)      RUN_PERF=1 ;;
                *) echo "unknown --only value: $2 (stability|speed|perf)" >&2; exit 1 ;;
            esac
            shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Build ir if needed
if [[ ! -f "target/release/ir" ]]; then
    echo "==> Building ir..."
    cargo build --release --bin ir 2>&1
fi
IR_BIN="$REPO_ROOT/target/release/ir"

FIXTURES_DIR="$REPO_ROOT/test-data/fixtures"
OVERALL=0   # 0=pass, 1=fail, 2=warn

_py() { python3 -c "$1"; }

monotonic_now() {
    python3 -c 'import time; print(time.monotonic())'
}

# Portable timeout: uses GNU timeout if present, else perl alarm (always on macOS).
_with_timeout() {
    local seconds="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "$seconds" "$@"
    else
        perl -e "alarm $seconds; exec @ARGV" -- "$@"
    fi
}

remove_collection() {
    "$IR_BIN" collection remove "$1" >/dev/null 2>&1 || true
}

run_prepare() {
    local fixture_dir="$1"
    local collection="$2"
    local preprocessor="$3"
    local embed_flag="$4"
    local args=(prepare --ir-bin "$IR_BIN" --data "$fixture_dir" --collection "$collection")
    [[ -n "$preprocessor" ]] && args+=(--preprocessor "$preprocessor")
    [[ -n "$embed_flag" ]] && args+=("$embed_flag")
    python3 scripts/beir-eval.py "${args[@]}"
}

# Assert a metric value: actual >= (expected - tolerance). Returns "PASS", "WARN", or "FAIL(...)".
assert_metric() {
    local actual="$1" floor="$2" calibrated="$3"
    python3 - <<EOF
actual = float("$actual")
floor = float("$floor")
if actual >= floor:
    print("PASS")
elif "$calibrated" != "True":
    print(f"WARN({actual:.3f}<{floor:.3f},uncal)")
else:
    print(f"FAIL({actual:.3f}<{floor:.3f})")
EOF
}

speed_probe() {
    local summary_path="$1" modes_csv="$2"
    python3 - "$summary_path" "$modes_csv" <<'EOF'
import json
import sys

summary_path, modes_csv = sys.argv[1:]
data = json.load(open(summary_path))
expected = set(m for m in modes_csv.split(",") if m)
seen = set()
missing = []
p50s = []

for result in data.get("results", []):
    mode = result.get("mode")
    if mode not in expected:
        continue
    seen.add(mode)
    p50 = (result.get("timing") or {}).get("median_ms")
    if p50 is None:
        missing.append(mode)
    else:
        p50s.append(float(p50))

missing.extend(sorted(expected - seen))
if missing:
    print("missing:" + ",".join(missing))
elif not p50s:
    print("missing:no-timing")
else:
    print("max:" + str(max(p50s)))
EOF
}

run_fixture() {
    local name="$1"
    local fixture_dir="$FIXTURES_DIR/$name"
    local exp="$fixture_dir/expected.json"

    if [[ ! -f "$fixture_dir/corpus.jsonl" ]]; then
        echo "[SKIP] $name — corpus not found (run scripts/generate-fixtures.sh)"
        return
    fi

    local collection_base="preship-$name"
    local collection_stability="${collection_base}-stability"
    local collection_measure="${collection_base}-measure"
    local max_wall calibrated
    max_wall=$(_py "import json; print(json.load(open('$exp'))['stability']['max_wall_clock_s'])")
    calibrated=$(_py "import json; print(json.load(open('$exp')).get('calibrated', False))")
    local preprocessor
    preprocessor=$(_py "import json; print(json.load(open('$exp')).get('preprocessor',''))" 2>/dev/null || echo "")

    local axis_parts=()
    local fixture_fail=0
    local fixture_warn=0

    # Determine which modes to check
    local run_modes=("bm25")
    if [[ "$BM25_ONLY" -eq 0 ]]; then
        _py "import json, sys; m=json.load(open('$exp')).get('performance',{}); sys.exit(0 if 'vector' in m else 1)" 2>/dev/null \
            && run_modes+=("vector") || true
        _py "import json, sys; m=json.load(open('$exp')).get('performance',{}); sys.exit(0 if 'hybrid' in m else 1)" 2>/dev/null \
            && run_modes+=("hybrid") || true
    fi
    local embed_flag=""
    [[ "$BM25_ONLY" -eq 0 ]] && [[ " ${run_modes[*]} " =~ " vector " || " ${run_modes[*]} " =~ " hybrid " ]] && embed_flag="--embed"

    # ── Stability ──────────────────────────────────────────────────────────────
    if [[ "$RUN_STABILITY" -eq 1 ]]; then
        local t0=$SECONDS
        local prep_status=0
        local prep_args=(prepare --ir-bin "$IR_BIN" --data "$fixture_dir" --collection "$collection_stability")
        [[ -n "$preprocessor" ]] && prep_args+=(--preprocessor "$preprocessor")
        [[ -n "$embed_flag" ]] && prep_args+=("$embed_flag")

        remove_collection "$collection_stability"
        _with_timeout "$max_wall" python3 scripts/beir-eval.py "${prep_args[@]}" >/dev/null 2>&1 || prep_status=$?
        local elapsed=$(( SECONDS - t0 ))

        if [[ "$prep_status" -eq 0 && "$elapsed" -le "$max_wall" ]]; then
            axis_parts+=("stability=PASS")
        elif [[ "$prep_status" -eq 124 || "$prep_status" -eq 142 || "$elapsed" -gt "$max_wall" ]]; then
            axis_parts+=("stability=FAIL(stall/timeout@${elapsed}s>${max_wall}s)")
            fixture_fail=1
            # Cleanup and skip speed/perf — collection state is unknown
            remove_collection "$collection_stability"
            local label="[FAIL]"
            echo "$label $name  ${axis_parts[*]}"
            [[ $OVERALL -lt 1 ]] && OVERALL=1
            return
        else
            axis_parts+=("stability=FAIL(exit=$prep_status)")
            fixture_fail=1
        fi

        remove_collection "$collection_stability"
    fi

    local summary_tmp=""
    if [[ "$RUN_SPEED" -eq 1 || "$RUN_PERF" -eq 1 ]]; then
        summary_tmp=$(mktemp "$TMPDIR/preship-XXXXXX")
        remove_collection "$collection_measure"

        local measure_prep_status=0
        local prep_started
        prep_started=$(monotonic_now)
        run_prepare "$fixture_dir" "$collection_measure" "$preprocessor" "$embed_flag" >/dev/null 2>&1 || measure_prep_status=$?
        local prep_finished
        prep_finished=$(monotonic_now)

        local summary_status=0
        local run_mode_csv
        run_mode_csv=$(IFS=,; echo "${run_modes[*]}")
        python3 scripts/beir-eval.py run \
            --ir-bin "$IR_BIN" \
            --data "$fixture_dir" \
            --collection "$collection_measure" \
            --mode "$run_mode_csv" \
            --at-k "10" \
            --output "$summary_tmp" >/dev/null 2>&1 || summary_status=$?

        if [[ "$RUN_SPEED" -eq 1 ]]; then
            local min_docs_per_s max_q_ms
            min_docs_per_s=$(_py "import json; print(json.load(open('$exp'))['speed']['min_index_docs_per_s'])")
            max_q_ms=$(_py "import json; print(json.load(open('$exp'))['speed']['max_query_p50_ms'])")

            local corpus_size
            corpus_size=$(wc -l < "$fixture_dir/corpus.jsonl" | tr -d ' ')
            local docs_per_s
            docs_per_s=$(_py "import math; elapsed=max(float('$prep_finished')-float('$prep_started'), 0.001); print(int(math.floor($corpus_size / elapsed)))")

            local speed_status
            local speed_probe_out
            speed_status=0
            speed_probe_out=$(speed_probe "$summary_tmp" "$run_mode_csv") || speed_status=$?

            local speed_fail=0
            if [[ "$measure_prep_status" -ne 0 ]]; then
                axis_parts+=("speed=FAIL(prepare-exit=$measure_prep_status)")
                speed_fail=1
            elif [[ "$summary_status" -ne 0 ]]; then
                axis_parts+=("speed=FAIL(run-exit=$summary_status)")
                speed_fail=1
            elif [[ "$speed_status" -ne 0 ]]; then
                axis_parts+=("speed=FAIL(p50-parse)")
                speed_fail=1
            elif [[ "$docs_per_s" -lt "$min_docs_per_s" ]]; then
                axis_parts+=("speed=FAIL(${docs_per_s}doc/s<${min_docs_per_s})")
                speed_fail=1
            elif [[ "$speed_probe_out" == missing:* ]]; then
                axis_parts+=("speed=FAIL(p50-missing:${speed_probe_out#missing:})")
                speed_fail=1
            else
                local query_p50_ms=${speed_probe_out#max:}
                local query_p50_int
                query_p50_int=$(_py "print(int(round(float('$query_p50_ms'))))" 2>/dev/null || echo "999999")
                if [[ "$query_p50_int" -le "$max_q_ms" ]]; then
                    axis_parts+=("speed=PASS(${docs_per_s}doc/s p50=${query_p50_int}ms)")
                else
                    axis_parts+=("speed=FAIL(p50=${query_p50_int}ms>${max_q_ms}ms)")
                    speed_fail=1
                fi
            fi

            if [[ "$speed_fail" -eq 1 ]]; then
                fixture_fail=1
            fi
        fi

        if [[ "$RUN_PERF" -eq 1 && "$summary_status" -eq 0 ]]; then
            for mode in "${run_modes[@]}"; do
                local ndcg recall floor_ndcg floor_recall
                ndcg=$(_py "import json; d=json.load(open('$summary_tmp')); r=[x for x in d['results'] if x['mode']=='$mode']; print(r[0]['metrics']['ndcg_10'] if r else 0)" 2>/dev/null || echo "0")
                recall=$(_py "import json; d=json.load(open('$summary_tmp')); r=[x for x in d['results'] if x['mode']=='$mode']; print(r[0]['metrics']['recall_10'] if r else 0)" 2>/dev/null || echo "0")
                floor_ndcg=$(_py "import json; e=json.load(open('$exp'))['performance']['$mode']; print(e['ndcg_10']-e['tolerance'])" 2>/dev/null || echo "0")
                floor_recall=$(_py "import json; e=json.load(open('$exp'))['performance']['$mode']; print(e['recall_10']-e['tolerance'])" 2>/dev/null || echo "0")
                # Per-mode uncalibrated flag overrides top-level calibrated
                mode_calibrated=$(_py "import json; e=json.load(open('$exp'))['performance']['$mode']; explicit=e.get('_uncalibrated'); print((not explicit) if explicit is not None else ('$calibrated' == 'True'))" 2>/dev/null || echo "False")

                local ndcg_result recall_result
                ndcg_result=$(assert_metric "$ndcg" "$floor_ndcg" "$mode_calibrated")
                recall_result=$(assert_metric "$recall" "$floor_recall" "$mode_calibrated")

                if [[ "$ndcg_result" == FAIL* || "$recall_result" == FAIL* ]]; then
                    axis_parts+=("perf-$mode=FAIL(ndcg=$ndcg_result recall=$recall_result)")
                    fixture_fail=1
                elif [[ "$ndcg_result" == WARN* || "$recall_result" == WARN* ]]; then
                    axis_parts+=("perf-$mode=WARN(ndcg=$ndcg_result recall=$recall_result)")
                    [[ $fixture_warn -eq 0 ]] && fixture_warn=1
                else
                    axis_parts+=("perf-${mode}=PASS(ndcg=${ndcg} recall=${recall})")
                fi
            done
        elif [[ "$RUN_PERF" -eq 1 ]]; then
            axis_parts+=("perf=FAIL(run-exit=$summary_status)")
            fixture_fail=1
        fi
    fi

    if [[ -n "$summary_tmp" ]]; then
        rm -f "$summary_tmp"
    fi
    remove_collection "$collection_measure"

    local label="[PASS]"
    if [[ $fixture_fail -eq 1 ]]; then
        label="[FAIL]"
        [[ $OVERALL -lt 1 ]] && OVERALL=1
    elif [[ $fixture_warn -eq 1 ]]; then
        label="[WARN]"
        [[ $OVERALL -lt 2 ]] && OVERALL=2
    fi
    echo "$label $name  ${axis_parts[*]}"
}

# ── Discover and run fixtures ─────────────────────────────────────────────────

echo "==> preship: building ir if needed..."
if [[ ! -f "target/release/ir" ]]; then
    cargo build --release --bin ir 2>&1
fi

echo "==> preship: running fixtures (bm25_only=$BM25_ONLY)"
echo ""

found=0
for fixture_dir in "$FIXTURES_DIR"/*/; do
    name=$(basename "$fixture_dir")
    [[ -n "$FIXTURE_FILTER" && "$name" != "$FIXTURE_FILTER" ]] && continue
    [[ ! -f "$fixture_dir/expected.json" ]] && continue
    found=1
    run_fixture "$name"
done

if [[ $found -eq 0 ]]; then
    echo "No fixtures found under $FIXTURES_DIR"
    exit 0
fi

echo ""
case $OVERALL in
    0) echo "==> PASS — all fixtures healthy" ;;
    1) echo "==> FAIL — one or more fixtures failed (see above)" ;;
    2) echo "==> WARN — perf drift detected; metrics below expected but above tolerance (run scripts/calibrate-fixtures.sh to update baselines)" ;;
esac

exit $OVERALL
