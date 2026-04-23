---
name: benchmark
description: "Benchmark ir search pipeline — design-first, parameterized experiments. Defines comparison matrix before running anything; finds best variant candidate before head-to-head. Use for threshold tuning, A/B comparison, signal collection, fusion weight sweeps."
argument-hint: "[what to measure]"
allowed-tools: Bash, Read, Write, Agent
---

# benchmark

**Core rule**: Design the experiment on paper before any run. Never blindly sweep — expensive runs only after cheap candidates are found.

**The harness is the factory.** Each run produces a row in a comparison table designed before any execution. Map axes (dataset × pool size × signal × threshold × seed) before deciding what to invoke. Parameterize first, produce outputs second.

See also: @~/.claude/skills/factory-philosophy/SKILL.md

---

## Phase 0: Pre-Ship Gate

Run this before any experiment. If it fails, the underlying `ir` build is suspect — results would be invalid.

```bash
bash scripts/preship.sh --bm25-only   # fast (~30s), skips embedding
bash scripts/preship.sh               # full, including vector + hybrid
```

Exit 0 = PASS. Exit 1 = FAIL (stop, investigate). Exit 2 = WARN (perf drift; note it and proceed).

**If preship fails:** see Phase 6 runbook.

---

## Phase 1: Design the Experiment Matrix

Before touching the shell, extract or ask for:

| Axis | Question |
|------|----------|
| **Goal** | What decision does this experiment inform? |
| **Baseline** | Fixed best-known config (what are we beating?) |
| **Variable(s)** | What changes? Range? Step size? |
| **Metric** | nDCG@10? recall@10? fire%? harm%? |
| **Corpora** | fiqa / miracl-ko / nfcorpus — which? |
| **Budget** | Full run vs sampled? How many queries? |

Print the matrix as a table and confirm before proceeding. If ARGUMENTS are ambiguous, ask.

**Example output:**

```
Goal:      Find best STRONG_SIGNAL_PRODUCT threshold for tier-1 shortcut
Baseline:  current HEAD (product=0.06, floor=0.40) — cached or run fresh
Variable:  STRONG_SIGNAL_PRODUCT ∈ [0.03, 0.05, 0.06, 0.08, 0.10]  (5 values)
           STRONG_SIGNAL_FLOOR   ∈ [0.35, 0.40, 0.45]               (3 values)
           → 15 combinations
Metric:    nDCG@10 (harm vs baseline), fire% (shortcut frequency)
Corpora:   fiqa, miracl-ko
Budget:    signal-sweep sampled (50000 docs for miracl-ko by default; 10000 remains the minimum stable floor) → shortlist → full bench top-3
```

---

## Phase 2: Corpus Check

```bash
ls test-data/{dataset}/corpus.jsonl test-data/{dataset}/queries.jsonl test-data/{dataset}/qrels/
```

Missing: `bash scripts/download-beir.sh {dataset}` or `bash scripts/download-miracl-ko.sh`.

| Shorthand | Dataset | Notes | Default pool size |
|-----------|---------|-------|-------------------|
| `synthetic-en` | committed fixture | 20 docs, deterministic; stability/perf canary | N/A (full) |
| `miracl-ko-mini` | committed fixture | 2k Korean docs; preprocessor deadlock canary | N/A (full) |
| `fiqa`, `en` | fiqa | English, general | per pool-size study |
| `ko`, `miracl-ko` | miracl-ko | Korean; BM25 near-zero → always hits tier 1+ | per pool-size study |
| `nfcorpus` | nfcorpus | Small, fast; good for cheap sweeps | N/A (small enough) |

Pool size recommendation: see `research/pool-size-study.md`. Current MIRACL-Ko research default: **50000 docs**. Minimum stable floor: **10000 docs**.
Do not use pool sizes `<= 503` for between-seed variance decisions: those samples collapse to the mandatory qrel-linked docs and are deterministic across seeds.

---

## Phase 3: Baseline Lock

Confirm or run the baseline once. Cache it.

```bash
# Check if baseline result is already cached
ls logs/results/{dataset}/{git7}.json

# If not cached:
bash scripts/research-harness.sh baseline --dataset fiqa
bash scripts/research-harness.sh baseline --dataset miracl-ko
```

If a crash happens after prepare/index/embed but before scoring completes, rerun the same `scripts/bench.sh {dataset}` command. The script resumes from a ready collection when possible instead of starting from zero. Query scoring itself also resumes: `scripts/beir-eval.py run --output ...` stores per-query progress in `<output>.partial/` and continues from there on rerun.
For non-BM25 runs, `scripts/bench.sh` pins tier-2 to the dedicated expander + reranker path and restarts the benchmark daemon before scoring. Do not manually start a daemon in combined mode and reuse it for benchmark runs.
For large corpora, `scripts/bench.sh {dataset} --size N --seed N` samples a BEIR-style pool on first run and benchmarks that sampled corpus directly. Use `miracl-ko --size 50000` as the default Korean sampled research baseline. Keep `10000` as the fast stable regression floor unless you explicitly need the full 1.5M-doc corpus.
On macOS, `scripts/bench.sh` also runs long phases under a safety watchdog by default. Metal stays on for speed, but the wrapper aborts the run if free memory drops too low, swapouts begin, or `ir` sustains CPU-fallback-like usage. Tune with `IR_BENCH_MIN_FREE_PCT`, `IR_BENCH_MAX_IR_CPU_PCT`, and `IR_BENCH_CPU_STRIKES`, or disable with `IR_BENCH_GUARD=0`.
For uncommitted working-tree changes, do not trust `scripts/bench.sh` cache keys by `HEAD` hash alone. If the change is query-time only (for example shortcut thresholds or rerank policy), reuse the prepared collection but write to a fresh output file such as `working-tree.json` and compare that against the hash-keyed baseline JSON. If the change affects indexing, preprocessing, or embeddings, rebuild a fresh collection instead of reusing the old one. After the quick A/B looks good, commit and rerun to produce the official hash-keyed artifact.

Baseline is a **fixed point** — do not re-run it per variant. All deltas are relative to this single run.

---

## Phase 4: Candidate Search (new method only)

If the variant has free parameters, find candidates cheaply before the full run.

**Rule**: Use the cheapest corpus + sampled queries to shortlist. Full benchmark only for top-3 candidates.

### Signal-based sweep (threshold tuning)

```bash
# Cheap: collect sampled signals and analyze threshold matrices
bash scripts/research-harness.sh thresholds --dataset fiqa
bash scripts/research-harness.sh thresholds --dataset miracl-ko --size 50000 --pools 3

# Then validate the shortlisted candidates on the locked holdout
bash scripts/research-harness.sh validate-thresholds --dataset fiqa
bash scripts/research-harness.sh validate-thresholds --dataset miracl-ko --size 50000 --pools 3
```

Candidate criteria: harm% < 5%, maximize fire%. Note FiQA vs MIRACL-Ko divergence — divergent = corpus-dependent, requires per-corpus constants.

Current maintainer read:

- keep FiQA on the current fused threshold unless holdout says otherwise
- use `miracl-ko --size 50000` as the Korean research corpus
- prefer stricter fused gating before trying a learned router
- keep router work offline until it clearly beats simple gating on holdout

### Holdout validation factory

`validate-thresholds` is the supported path for threshold tuning:

1. shortlist candidates from existing threshold sweep JSON
2. reuse the locked holdout collection when the change is query-time only
3. run each candidate via env overrides, not source edits
4. print a baseline-vs-candidate table on the holdout

This keeps production defaults unchanged until a candidate survives real validation.
For a fine sweep near a boundary, bypass shortlist generation and validate exact fused values:

```bash
bash scripts/research-harness.sh validate-thresholds \
  --dataset miracl-ko --size 50000 --gate fused \
  --products 0.0525,0.055,0.0575,0.06
```

### Code-constant sweep (fusion weights, floor/product)

Sweeping code constants requires rebuild per value — expensive. Strategy:

1. Use signal data to estimate effect *before* rebuilding (`threshold-sweep.py` can simulate)
2. Validate threshold-like changes with `validate-thresholds` first
3. Only patch source for the winner or for non-env-tunable constants

```bash
# Only after the validation factory has identified a winner
# Edit src/search/hybrid.rs
cargo build --release --bin ir
scripts/bench.sh {dataset}
```

### Env-var sweep (no rebuild needed)

```bash
IR_DISABLE_SHORTCUTS=1 scripts/bench.sh {dataset}
# or other env vars — no rebuild, cheap to sweep
```

---

## Phase 5: Head-to-Head Comparison

Baseline best-known vs variant best candidate(s).

```bash
# Compare HEAD vs a prior git ref
scripts/bench.sh {dataset} {baseline-git7}
```

Report format — print this table for every comparison:

```
Dataset: fiqa
Metric:  nDCG@10

Config                      | nDCG@10 | recall@10 | fire% | harm%  | Δ nDCG
----------------------------|---------|-----------|-------|--------|-------
baseline (product=0.06)     | 0.412   | 0.531     | 34%   | —      | —
variant A (product=0.05)    | 0.418   | 0.537     | 28%   | 0.8%   | +0.006
variant B (product=0.08)    | 0.409   | 0.528     | 41%   | 2.1%   | -0.003

Winner: variant A — higher nDCG, lower fire%, harm within budget
```

For a query-time-only working-tree validation, the comparison is:

```bash
# baseline artifact (existing)
logs/results/{dataset}/{git7}.json

# working-tree artifact (fresh output path, same data + same collection)
logs/results/{dataset}/working-tree.json
```

This is valid only when the indexed collection itself is unchanged. If preprocessing, indexing, or embeddings changed, the collection must be rebuilt first.

---

## Phase 6: Apply + Verify

```bash
# Update src/search/hybrid.rs constants
# BM25_STRONG_FLOOR / BM25_STRONG_GAP         (tier-0)
# STRONG_SIGNAL_FLOOR / STRONG_SIGNAL_PRODUCT  (tier-1)

cargo test
scripts/bench.sh fiqa
scripts/bench.sh miracl-ko
```

No regression = both corpora nDCG within ±0.005 of baseline.

---

## Phase 7: Progress Monitoring + Canary Failure Runbook

After starting any long-running prepare or sweep, check `logs/sweep-runs/*.json` for timing summaries. Timestamps on all shell echoes (format `[HH:MM:SS]`). The indexer progress bar shows `{per_sec}` — expected ≥50 docs/s for small corpora.

**Stability FAIL (stall pulse in indexer, or preship stability=FAIL)**
Almost certainly upstream issue #13 — all-filtered-line lindera deadlock in `src/preprocess.rs`. See `knowledge/sessions/ir-bench/session-20260418-1528.md` for full root cause and three candidate fixes.
- Kill the process immediately.
- Re-run `scripts/preship.sh --fixture miracl-ko-mini` — if it also stalls, the deadlock is reproducible on the 2k canary.
- Report to upstream issue #13 with the failing passage if found.

**Speed FAIL (docs/sec below budget)**
Likely embedder on CPU. Check: `IR_FORCE_CPU_BACKEND` unset, GPU layers loaded (default `IR_GPU_LAYERS=99`). If preprocessor: throughput drop → bisect on `src/preprocess.rs`.

**Performance FAIL (metric drift > tolerance)**
Real retrieval-quality regression. Bisect on `src/search/`, `src/llm/`, `src/index/embed.rs`. Do NOT relax tolerance to make assertions green — investigate root cause.

---

## Reusable Components

Each phase is independent. Re-compose for new research questions:

| Component | Reuse for |
|-----------|-----------|
| `scripts/research-harness.sh` | Maintainer entrypoint for baseline, signals, thresholds |
| `scripts/threshold-validate.py` | Shortlist stable candidates from sweep JSON for holdout runs |
| Pre-ship gate (Phase 0) | Every experiment — gate on this before any run |
| `scripts/preship.sh` | Stability/speed/performance pre-ship check |
| `test-data/fixtures/synthetic-en/` | English BM25/vector/hybrid canary |
| `test-data/fixtures/miracl-ko-mini/` | Korean preprocessor deadlock canary |
| `scripts/generate-fixtures.sh` | Populate miracl-ko-mini from downloaded corpus |
| `scripts/calibrate-fixtures.sh` | Update expected.json after pipeline change |
| `research/pool-size-study.md` | Pool size defaults (run once, reference often) |
| Corpus check (Phase 2) | Any new experiment |
| Baseline lock (Phase 3) | Any A/B — always run once first |
| Signal sweep (Phase 4) | Any threshold research — cheap, no rebuild |
| Env-var sweep (Phase 4) | Any flag/env research — no rebuild |
| Holdout validation factory (Phase 4) | Threshold tuning with env overrides, no rebuild per candidate |
| Code-constant sweep (Phase 4) | Fusion weights and other non-env-tunable constants — rebuild required |
| Comparison table (Phase 5) | Any comparison — standard report format |
| Regression verify (Phase 6) | Any change to `src/search/hybrid.rs` |
| Progress runbook (Phase 7) | Any hang or metric anomaly during sweep |

When a new benchmark question arrives: **run preship.sh first** → name the goal → pick components → design matrix → run only what's needed.
