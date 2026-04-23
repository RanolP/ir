# ir — Research Harness

Current benchmark and threshold-tuning workflow for maintainers.

This document replaces the older ad hoc experiment log. The source of truth is now:

- `scripts/research-harness.sh`
- `scripts/bench.sh`
- `scripts/signal-sweep.sh`
- `scripts/threshold-sweep.py`
- `scripts/threshold-validate.py`
- `research/pool-size-study.md`

## Purpose

Use this harness for:

- baseline locking
- threshold research for the tier-0 BM25 gate
- threshold research for the tier-1 fused gate
- branch-to-branch benchmark comparisons

Do not use sampled MIRACL-Ko scores as full-corpus quality claims. They are for relative regression and A/B decisions only.

## Current Baselines

Captured on `HEAD=4acefe9`.

### FiQA Full Corpus

Dataset:
- `57,638` docs
- `648` benchmark queries with qrels

| Mode | nDCG@10 | R@10 | med ms |
|---|---:|---:|---:|
| bm25 | 0.2447 | 0.3003 | 209.3 |
| vector | 0.4045 | 0.4932 | 222.9 |
| hybrid | 0.4431 | 0.5371 | 5041.6 |

Interpretation:
- quality baseline is credible
- hybrid quality gain over vector is real
- hybrid latency is still high because unique-query benchmark runs pay tier-2 cost repeatedly

### MIRACL-Ko Sampled Pool

Dataset:
- `miracl-ko-s50000-p42`
- `50,000` docs sampled from the full `1,486,752`-doc corpus
- `213` benchmark queries

| Mode | nDCG@10 | R@10 | med ms |
|---|---:|---:|---:|
| bm25 | 0.7271 | 0.8130 | 102.1 |
| vector | 0.9109 | 0.9426 | 168.8 |
| hybrid | 0.9630 | 0.9813 | 1131.9 |

Interpretation:
- credible as a sampled-pool regression baseline
- not a full-corpus absolute score
- use for branch comparison and threshold selection, not for public full-MIRACL claims

## Current Conclusions Before Runtime Changes

- `fiqa`: keep the current fused strong-signal product at `0.06`
- `miracl-ko --size 50000`: holdout validation on `p42` currently favors stricter fused gating at `0.05` over `0.06`
  - `0.05`: `nDCG@10=0.9650`, `R@10=0.9813`, `med=431.5ms`
  - `0.06`: `nDCG@10=0.9603`, `R@10=0.9766`, `med=440.4ms`
- BM25 thresholds are not the main lever on either corpus
- router work stays offline research for now
  - the mixed router is not good enough on FiQA
  - the Korean-only router is more promising, but `tier1` already matched `hybrid` on the current `miracl-ko-s50000-p42` holdout
- the next runtime change should be threshold/config driven, not router driven
- `hybrid` already behaves as "BM25 while cold, fused once warm"
  - cold first query can return BM25 immediately while the daemon warms
  - later warm `hybrid` queries go through tier-1 fused retrieval before tier-2

## Research Workflow

### 1. Gate the Build

Always run the fast regression gate first:

```bash
bash scripts/preship.sh --bm25-only
```

If this fails, benchmark results are not trustworthy.

### 2. Lock a Baseline

Recommended commands:

```bash
bash scripts/research-harness.sh baseline --dataset fiqa
bash scripts/research-harness.sh baseline --dataset miracl-ko
```

Notes:
- `baseline --dataset miracl-ko` defaults to `--size 50000`
- non-BM25 benchmark runs pin tier-2 to the dedicated expander + reranker path
- reruns resume from prepared collections and per-query sidecars when present

### 3. Collect Signal Data

FiQA:

```bash
bash scripts/research-harness.sh signals --dataset fiqa
```

MIRACL-Ko sampled pools:

```bash
bash scripts/research-harness.sh signals --dataset miracl-ko --size 50000 --pools 3
```

Notes:
- `signals --dataset miracl-ko` defaults to sampled `--size 50000 --pools 3`
- sampled signal runs use `signal-sweep.sh --sample-only`
- this avoids dragging the full `1.5M`-doc corpus into threshold research

### 4. Sweep the Threshold Matrix

FiQA:

```bash
bash scripts/research-harness.sh thresholds --dataset fiqa
```

MIRACL-Ko sampled pools:

```bash
bash scripts/research-harness.sh thresholds --dataset miracl-ko --size 50000 --pools 3
```

Output:

- console tables from `scripts/threshold-sweep.py`
- machine-readable JSON under `.bench-state/research/*.json`

To re-analyze existing signals without recollecting:

```bash
bash scripts/research-harness.sh thresholds --dataset miracl-ko --size 50000 --pools 3 --analyze-only
```

### 5. Validate the Shortlist on Holdout

Do not patch source just to test one candidate at a time. The supported flow is:

```bash
bash scripts/research-harness.sh validate-thresholds --dataset fiqa
bash scripts/research-harness.sh validate-thresholds --dataset miracl-ko --size 50000 --pools 3
```

What this does:

1. loads the existing threshold sweep JSON
2. shortlists candidates that pass the harm/fire budget
3. reuses the locked holdout collection when possible
4. runs each candidate through env overrides
5. prints a baseline-vs-candidate holdout table

For a fine sweep near a known boundary, skip the offline shortlist and validate exact values directly:

```bash
bash scripts/research-harness.sh validate-thresholds \
  --dataset miracl-ko --size 50000 --gate fused \
  --products 0.0525,0.055,0.0575,0.06
```

## Thresholds Under Study

Current code constants in `src/search/hybrid.rs`:

| Gate | Constant | Current |
|---|---|---:|
| Tier-0 BM25 | `BM25_STRONG_FLOOR` | 0.75 |
| Tier-0 BM25 | `BM25_STRONG_GAP` | 0.10 |
| Tier-1 fused | `STRONG_SIGNAL_FLOOR` | 0.40 |
| Tier-1 fused | `STRONG_SIGNAL_PRODUCT` | 0.06 |

Current sweep grids:

### Tier-0 BM25

- floor: `0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90`
- gap: `0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20`

### Tier-1 Fused

- product: `0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15`

The offline analyzer currently treats fused-floor as fixed and sweeps product only.

## Selection Rule

Choose candidate thresholds by:

1. `harm% < 5%`
2. highest `fire%` among those candidates
3. same directional behavior across:
   - `fiqa`
   - `miracl-ko-s50000-p{1..N}`

Do not promote a candidate from one corpus alone unless the change is intentionally corpus-specific.

## Validation Loop

After selecting candidates:

1. run `validate-thresholds`
2. keep only the holdout winner
3. patch source for that winner only
4. rerun the official baseline so the final artifact is hash-keyed

Only after that should you consider a full-corpus MIRACL-Ko run.

## Optional: Offline Tier-2 Router Research

This is not the current implementation path.

Use it only to answer:

- can a learned router beat simple stricter gating on holdout?
- is there corpus-specific signal that fixed thresholds are missing?

Signal sweeps can be exported into smoltrain JSONL for a binary `skip_tier2` vs `run_tier2` classifier:

```bash
bash scripts/signal-sweep.sh --dataset miracl-ko --size 50000 --pools 3 --tier1
python3 scripts/export-tier2-router-data.py \
  logs/signals/miracl-ko-s50000-p1 \
  logs/signals/miracl-ko-s50000-p2 \
  logs/signals/miracl-ko-s50000-p3 \
  --output .bench-state/research/tier2-router-ko-s50000.jsonl
```

To keep router work isolated from the main benchmark harness, use the dedicated wrapper:

```bash
bash scripts/router-data.sh ko
bash scripts/router-data.sh fiqa
bash scripts/router-data.sh mixed
```

Current recommendation:

- use `ko` first
- do not ship the mixed router yet; its FiQA `run_tier2` recall is still too weak
- do not wire any router into runtime until it clearly beats stricter fused gating on holdout

The exporter uses only query-time features and now writes both:

- `text` for current `smoltrain`
- `input` for backward compatibility with older notes

To make a self-contained `smoltrain` bundle:

```bash
python3 scripts/prepare-tier2-router-smoltrain.py \
  --input .bench-state/research/tier2-router.jsonl \
  --output-dir .bench-state/research/tier2-router-smoltrain
```

The wrapper above already does both export + bundle prep.

Then train from that directory:

```bash
cd .bench-state/research/tier2-router-smoltrain
PYTHONPATH=/Users/eliot/ws-ps/smoltrain python3 -m smoltrain.train \
  --data train_balanced.jsonl --taxonomy taxonomy.yaml --epochs 10 --seed 42
PYTHONPATH=/Users/eliot/ws-ps/smoltrain python3 -m smoltrain.eval \
  --model models/charcnn_trained.onnx --data train.jsonl \
  --taxonomy taxonomy.yaml --world world.json --eval-data eval.jsonl
```

The exported features are:

- query text and simple language/question heuristics
- BM25 top/gap and top-10 score shape
- fused top/gap

Labels come from held-out benchmark behavior:

- `run_tier2` when hybrid materially beats tier-1 fused on that query
- `skip_tier2` otherwise

To benchmark a trained checkpoint on a holdout without changing `ir`:

```bash
IR_CONFIG_DIR=.bench-state/bench/xdg/ir TMPDIR=.bench-state/bench/tmp \
python3 scripts/beir-eval.py run \
  --ir-bin target/release/ir \
  --data test-data/miracl-ko-s50000-p42 \
  --collection eval-miracl-ko-s50000-p42-4acefe9 \
  --mode bm25,vector,tier1,hybrid \
  --signals \
  --signals-output logs/signals/miracl-ko-s50000-p42

/Users/eliot/ws-ps/smoltrain/.venv/bin/python scripts/router-bench.py \
  --signals logs/signals/miracl-ko-s50000-p42 \
  --checkpoint .bench-state/research/tier2-router-ko-s50000-smoltrain/models/charcnn_trained.pt \
  --thresholds 0.3,0.4,0.5
```

Current read:

- useful as an offline research tool
- not yet justified as the next runtime change

## Known Caveats

### Sampled MIRACL-Ko Is Easier Than Full MIRACL-Ko

The sampler always keeps all qrel-linked docs and fills the rest with sampled negatives.

That makes:

- absolute scores inflated
- relative branch-to-branch comparisons still useful

See `research/pool-size-study.md` for why `10000` docs is still the minimum stable pool, even though the active research default is now `50000` for better metric headroom.

### Hybrid Benchmark Latency Is Not Interactive Latency

The benchmark uses many unique queries, so caches help less than they do in repeated interactive use.

High hybrid median latency means:

- tier-2 expansion + rerank work dominates
- not that daemon startup is still happening on every query

### Full MIRACL-Ko Is Still Expensive

`src/index/embed.rs` no longer preloads the full pending corpus into memory before starting progress, but the full `1.5M`-doc Korean corpus is still expensive enough that the default research path should remain the sampled `50k` pool.

## Output Locations

Baselines:

- `logs/results/fiqa/<git7>.json`
- `logs/results/miracl-ko-s50000-p42/<git7>.json`

Signal collections:

- `logs/signals/fiqa/`
- `logs/signals/miracl-ko-s50000-p1/`
- `logs/signals/miracl-ko-s50000-p2/`
- `logs/signals/miracl-ko-s50000-p3/`

Threshold analyses:

- `.bench-state/research/fiqa-thresholds.json`
- `.bench-state/research/miracl-ko-s50000-p3-thresholds.json`

Validation outputs:

- `.bench-state/research/fiqa-fused-candidates.json`
- `.bench-state/research/miracl-ko-s50000-fused-candidates.json`
- `.bench-state/research/validate/<dataset>/`

## Recommended Defaults

Use these unless you have a specific reason not to:

- baseline English corpus: `fiqa`
- baseline Korean corpus: `miracl-ko --size 50000`
- threshold research Korean pool: `50000` docs
- threshold research Korean seeds: `3`
- regression gate: `bash scripts/preship.sh --bm25-only`
