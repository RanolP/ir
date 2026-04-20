#!/usr/bin/env bash
# calibrate-fixtures.sh — Measure actual metrics for a fixture and update expected.json.
#
# Run once after generating a new fixture (or after a deliberate pipeline change) to
# lock in the baseline. Sets calibrated=true and tightens tolerances to 10% below measured.
#
# Usage:
#   scripts/calibrate-fixtures.sh synthetic-en
#   scripts/calibrate-fixtures.sh miracl-ko-mini
#   scripts/calibrate-fixtures.sh synthetic-en --bm25-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "calibrate-fixtures"

NAME="${1:-}"
BM25_ONLY=0
[[ "${2:-}" == "--bm25-only" ]] && BM25_ONLY=1

if [[ -z "$NAME" ]]; then
    echo "Usage: scripts/calibrate-fixtures.sh <fixture-name> [--bm25-only]" >&2
    exit 1
fi

FIXTURE_DIR="$REPO_ROOT/test-data/fixtures/$NAME"
EXP="$FIXTURE_DIR/expected.json"

if [[ ! -f "$FIXTURE_DIR/corpus.jsonl" ]]; then
    echo "ERROR: $FIXTURE_DIR/corpus.jsonl not found" >&2
    exit 1
fi

if [[ ! -f "target/release/ir" ]]; then
    cargo build --release --bin ir 2>&1
fi
IR_BIN="$REPO_ROOT/target/release/ir"
COLLECTION="calibrate-$NAME"

echo "==> Calibrating $NAME..."

python3 - "$FIXTURE_DIR" "$EXP" "$IR_BIN" "$COLLECTION" "$BM25_ONLY" <<'PYEOF'
import json, subprocess, sys, os, tempfile, shutil

fixture_dir, exp_path, ir_bin, collection, bm25_only_str = sys.argv[1:]
bm25_only = bm25_only_str == "1"
fixture_dir = os.path.abspath(fixture_dir)
exp_path = os.path.abspath(exp_path)

e = json.load(open(exp_path))
expected_modes = list(e.get("performance", {}).keys())
if bm25_only:
    expected_modes = [m for m in expected_modes if m == "bm25"] or ["bm25"]

preprocessor = e.get("preprocessor", "")
embed = (not bm25_only) and any(m in ("vector", "hybrid") for m in expected_modes)

def run(cmd, **kw):
    print(" ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)

# Prepare
print("  preparing collection...")
prep = [ir_bin, "collection", "add", collection, os.path.join(fixture_dir, "eval-corpus"),
        "--glob", "**/*.txt"]
if preprocessor:
    prep += ["--preprocessor", preprocessor]

# Use beir-eval.py prepare which handles the full flow
script = os.path.join(os.path.dirname(ir_bin), "..", "..", "scripts", "beir-eval.py")
prep_args = ["python3", script, "prepare",
             "--ir-bin", ir_bin, "--data", fixture_dir, "--collection", collection]
if preprocessor:
    prep_args += ["--preprocessor", preprocessor]
if embed:
    prep_args.append("--embed")
run(prep_args)

# Measure each mode
results = {}
for mode in expected_modes:
    print(f"  measuring {mode}...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                     dir=os.environ.get("TMPDIR", "/tmp")) as tf:
        tmpout = tf.name
    run_args = ["python3", script, "run",
                "--ir-bin", ir_bin, "--data", fixture_dir, "--collection", collection,
                "--mode", mode, "--at-k", "10", "--output", tmpout]
    run(run_args)
    d = json.load(open(tmpout))
    os.unlink(tmpout)
    r = [x for x in d["results"] if x["mode"] == mode]
    if r:
        results[mode] = {"ndcg_10": r[0]["metrics"]["ndcg_10"],
                         "recall_10": r[0]["metrics"]["recall_10"]}
        print(f"    {mode}: nDCG@10={results[mode]['ndcg_10']} Recall@10={results[mode]['recall_10']}")
    else:
        print(f"    {mode}: no results (check collection)")

# Cleanup collection
subprocess.run([ir_bin, "collection", "remove", collection],
               capture_output=True)

# Update expected.json
TOLERANCE = 0.03
BUFFER    = 0.10  # floor = measured * (1 - BUFFER)

for mode, vals in results.items():
    e["performance"][mode] = {
        "ndcg_10":   round(vals["ndcg_10"]   * (1 - BUFFER), 3),
        "recall_10": round(vals["recall_10"] * (1 - BUFFER), 3),
        "tolerance": TOLERANCE,
    }

all_measured = all(m in results for m in expected_modes)
if all_measured:
    e["calibrated"] = True
    e.pop("_note", None)
else:
    unmeasured = [m for m in expected_modes if m not in results]
    e["_note"] = f"Partially calibrated. Unmeasured modes: {unmeasured}. Re-run without --bm25-only to fully calibrate."

with open(exp_path, "w") as f:
    json.dump(e, f, indent=2)
    f.write("\n")

print(f"\nUpdated {exp_path} (calibrated=true)")
print("Review and commit.")
PYEOF

# Clean up eval-corpus dir created during prepare (not committed)
rm -rf "$FIXTURE_DIR/eval-corpus"
