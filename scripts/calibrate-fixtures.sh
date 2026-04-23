#!/usr/bin/env bash
# calibrate-fixtures.sh — Measure actual metrics for a fixture and update expected.json.
#
# Run once after generating a new fixture (or after a deliberate pipeline change) to
# lock in the baseline. Performance floors are set to 10% below the worst observed run,
# and query-p50 ceilings are set to 10% above the slowest observed run.
#
# Usage:
#   scripts/calibrate-fixtures.sh synthetic-en
#   scripts/calibrate-fixtures.sh miracl-ko-mini
#   scripts/calibrate-fixtures.sh synthetic-en --bm25-only
#   scripts/calibrate-fixtures.sh synthetic-en --runs 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "calibrate-fixtures"

NAME="${1:-}"
BM25_ONLY=0
RUNS=3

shift_count=0
if [[ -n "$NAME" ]]; then
    shift_count=1
fi
if [[ "$shift_count" -eq 1 ]]; then
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bm25-only) BM25_ONLY=1; shift ;;
        --runs) RUNS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$NAME" ]]; then
    echo "Usage: scripts/calibrate-fixtures.sh <fixture-name> [--bm25-only] [--runs N]" >&2
    exit 1
fi

if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid --runs '$RUNS' (expected positive integer)" >&2
    exit 1
fi

FIXTURE_DIR="$REPO_ROOT/test-data/fixtures/$NAME"
EXP="$FIXTURE_DIR/expected.json"
EVAL_CORPUS_DIR="$FIXTURE_DIR/eval-corpus"
EVAL_CORPUS_PREEXISTED=0
if [[ -d "$EVAL_CORPUS_DIR" ]]; then
    EVAL_CORPUS_PREEXISTED=1
fi

if [[ ! -f "$FIXTURE_DIR/corpus.jsonl" ]]; then
    echo "ERROR: $FIXTURE_DIR/corpus.jsonl not found" >&2
    exit 1
fi

if [[ ! -f "target/release/ir" ]]; then
    cargo build --release --bin ir 2>&1
fi
IR_BIN="$REPO_ROOT/target/release/ir"
COLLECTION="calibrate-$NAME"

echo "==> Calibrating $NAME (runs=$RUNS bm25_only=$BM25_ONLY)..."

python3 - "$FIXTURE_DIR" "$EXP" "$IR_BIN" "$COLLECTION" "$BM25_ONLY" "$RUNS" <<'PYEOF'
import json, math, os, subprocess, sys, tempfile, time

fixture_dir, exp_path, ir_bin, collection, bm25_only_str, runs_str = sys.argv[1:]
bm25_only = bm25_only_str == "1"
runs = int(runs_str)
fixture_dir = os.path.abspath(fixture_dir)
exp_path = os.path.abspath(exp_path)

e = json.load(open(exp_path))
all_perf_modes = list(e.get("performance", {}).keys())
expected_modes = list(all_perf_modes)
if bm25_only:
    expected_modes = [m for m in expected_modes if m == "bm25"] or ["bm25"]

preprocessor = e.get("preprocessor", "")
embed = (not bm25_only) and any(m in ("vector", "hybrid") for m in expected_modes)
corpus_size = sum(1 for line in open(os.path.join(fixture_dir, "corpus.jsonl")) if line.strip())
tmp_dir = os.environ.get("TMPDIR", fixture_dir)

def run(cmd, **kw):
    print(" ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)

script = os.path.join(os.path.dirname(ir_bin), "..", "..", "scripts", "beir-eval.py")
run_mode_csv = ",".join(expected_modes)
perf_runs = {mode: {"ndcg_10": [], "recall_10": []} for mode in expected_modes}
docs_per_s_runs = []
p50_runs = []
all_measured = True
full_mode_calibration = set(expected_modes) == set(all_perf_modes)

for run_idx in range(1, runs + 1):
    collection_name = f"{collection}-r{run_idx}"
    print(f"\n==> run {run_idx}/{runs}: prepare {collection_name}")
    prep_args = ["python3", script, "prepare",
                 "--ir-bin", ir_bin, "--data", fixture_dir, "--collection", collection_name]
    if preprocessor:
        prep_args += ["--preprocessor", preprocessor]
    if embed:
        prep_args.append("--embed")

    t0 = time.monotonic()
    run(prep_args)
    prep_elapsed = max(time.monotonic() - t0, 0.001)
    docs_per_s = corpus_size / prep_elapsed
    docs_per_s_runs.append(docs_per_s)
    print(f"    prepare: {prep_elapsed:.3f}s ({docs_per_s:.1f} doc/s)")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=tmp_dir) as tf:
        tmpout = tf.name
    try:
        print(f"==> run {run_idx}/{runs}: query modes {run_mode_csv}")
        run_args = ["python3", script, "run",
                    "--ir-bin", ir_bin, "--data", fixture_dir, "--collection", collection_name,
                    "--mode", run_mode_csv, "--at-k", "10", "--output", tmpout]
        run(run_args)
        d = json.load(open(tmpout))
        run_results = {x["mode"]: x for x in d.get("results", [])}
        for mode in expected_modes:
            result = run_results.get(mode)
            if not result:
                print(f"    {mode}: no results")
                all_measured = False
                continue
            metrics = result.get("metrics", {})
            timing = result.get("timing", {})
            ndcg = float(metrics["ndcg_10"])
            recall = float(metrics["recall_10"])
            perf_runs[mode]["ndcg_10"].append(ndcg)
            perf_runs[mode]["recall_10"].append(recall)
            p50 = timing.get("median_ms")
            if p50 is not None:
                p50_runs.append(float(p50))
            print(f"    {mode}: nDCG@10={ndcg:.4f} Recall@10={recall:.4f} p50={p50}")
    finally:
        if os.path.exists(tmpout):
            os.unlink(tmpout)
        subprocess.run([ir_bin, "collection", "remove", collection_name], capture_output=True)

TOLERANCE = 0.03
BUFFER = 0.10

for mode, vals in perf_runs.items():
    if not vals["ndcg_10"] or not vals["recall_10"]:
        all_measured = False
        continue
    e["performance"][mode] = {
        "ndcg_10": round(min(vals["ndcg_10"]) * (1 - BUFFER), 3),
        "recall_10": round(min(vals["recall_10"]) * (1 - BUFFER), 3),
        "tolerance": TOLERANCE,
        "_uncalibrated": False,
    }

for mode in all_perf_modes:
    if mode not in expected_modes:
        e["performance"].setdefault(mode, {})
        e["performance"][mode]["_uncalibrated"] = True
    elif full_mode_calibration:
        e["performance"][mode].pop("_uncalibrated", None)

if docs_per_s_runs:
    e.setdefault("speed", {})
    e["speed"]["min_index_docs_per_s"] = max(1, int(math.floor(min(docs_per_s_runs) * (1 - BUFFER))))
if p50_runs:
    e.setdefault("speed", {})
    e["speed"]["max_query_p50_ms"] = max(1, int(math.ceil(max(p50_runs) * (1 + BUFFER))))

if all_measured and p50_runs and docs_per_s_runs and full_mode_calibration:
    e["calibrated"] = True
    e.pop("_note", None)
else:
    unmeasured = [m for m in expected_modes if not perf_runs[m]["ndcg_10"]]
    skipped = [m for m in all_perf_modes if m not in expected_modes]
    note_parts = []
    if unmeasured:
        note_parts.append(f"Unmeasured modes: {unmeasured}.")
    if skipped:
        note_parts.append(f"Skipped modes: {skipped}.")
    note_parts.append("Re-run without --bm25-only to fully calibrate.")
    e["calibrated"] = False
    e["_note"] = (
        "Partially calibrated. " + " ".join(note_parts)
    )

with open(exp_path, "w") as f:
    json.dump(e, f, indent=2)
    f.write("\n")

print(f"\nUpdated {exp_path}")
if docs_per_s_runs:
    print(f"  speed floor: {e['speed']['min_index_docs_per_s']} doc/s "
          f"(from worst run {min(docs_per_s_runs):.1f} doc/s)")
if p50_runs:
    print(f"  query p50 ceiling: {e['speed']['max_query_p50_ms']} ms "
          f"(from slowest run {max(p50_runs):.1f} ms)")
print("Review and commit.")
PYEOF

# Clean up only if calibration had to create a throwaway eval-corpus dir.
if [[ "$EVAL_CORPUS_PREEXISTED" -eq 0 ]]; then
    rm -rf "$EVAL_CORPUS_DIR"
fi
