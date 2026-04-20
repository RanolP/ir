#!/usr/bin/env bash
# generate-fixtures.sh — Populate committed fixtures that require downloading large source datasets.
#
# Currently populates:
#   test-data/fixtures/miracl-ko-mini/  (2000-doc deterministic sample of MIRACL-Ko, seed=42)
#
# Requires: scripts/download-miracl-ko.sh to have been run (or MIRACL-Ko corpus already present)
# Usage:    scripts/generate-fixtures.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "generate-fixtures"

FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── miracl-ko-mini ────────────────────────────────────────────────────────────

FIXTURE="$REPO_ROOT/test-data/fixtures/miracl-ko-mini"
SOURCE="$REPO_ROOT/test-data/miracl-ko"

if [[ -f "$FIXTURE/corpus.jsonl" && "$FORCE" -eq 0 ]]; then
    echo "[skip] test-data/fixtures/miracl-ko-mini — already populated (use --force to regenerate)"
else
    if [[ ! -f "$SOURCE/corpus.jsonl" ]]; then
        echo "ERROR: MIRACL-Ko corpus not found at $SOURCE/corpus.jsonl" >&2
        echo "       Run: bash scripts/download-miracl-ko.sh" >&2
        exit 1
    fi

    echo "==> Generating miracl-ko-mini (2000 docs, seed=42)..."
    # Remove existing generated files but preserve expected.json placeholder.
    rm -f "$FIXTURE/corpus.jsonl" "$FIXTURE/queries.jsonl"
    rm -rf "$FIXTURE/qrels"
    mkdir -p "$FIXTURE"

    stage_dir=$(mktemp -d "$TMPDIR/miracl-ko-mini-XXXXXX")

    python3 scripts/beir-eval.py sample \
        --data "$SOURCE" \
        --size 2000 \
        --seed 42 \
        --output "$stage_dir"

    mv "$stage_dir/corpus.jsonl" "$FIXTURE/corpus.jsonl"
    mv "$stage_dir/queries.jsonl" "$FIXTURE/queries.jsonl"
    mv "$stage_dir/qrels" "$FIXTURE/qrels"
    rmdir "$stage_dir"

    echo "==> Done. Fixture at test-data/fixtures/miracl-ko-mini/"
    echo "    Commit corpus.jsonl, queries.jsonl, qrels/ alongside expected.json."
    echo "    Then run: scripts/calibrate-fixtures.sh miracl-ko-mini"
fi
