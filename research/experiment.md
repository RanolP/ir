# ir — Research & Experiments

Ongoing benchmark results and model experiments.
Baseline system: EmbeddingGemma 308M + Qwen3-Reranker 0.6B + qmd-expansion 1.7B.

## Benchmark Setup

**Dataset**: BEIR/NFCorpus — 3,633 medical documents · 323 test queries · graded relevance.
**Metric**: nDCG@10 (primary), Recall@10 (secondary).

```bash
# Download dataset (~100MB)
curl -L https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip \
  -o /tmp/nfcorpus.zip && unzip /tmp/nfcorpus.zip -d test-data/

cargo run --release --bin eval -- --data test-data/nfcorpus --mode all
```

## Baseline Results (NFCorpus)

| Mode | nDCG@10 | Recall@10 | Notes |
|------|---------|-----------|-------|
| BM25 | 0.2046 | 0.0932 | no model |
| Vector | 0.3898 | 0.1926 | EmbeddingGemma 300M |
| **Hybrid (score-fusion α=0.80)** | **0.3954** | **0.1958** | +1.4% vs vector |
| Hybrid + reranker | 0.4001 | — | +1.2% vs score-fusion |

Old pure-RRF scored 0.372 — score-fusion is +5.5% better.

---

## Experiment: Alpha Sensitivity (α=0.80 vs α=0.95)

**Question**: Does pushing toward pure vector (α=0.95) improve results over α=0.80?

```bash
for ds in nfcorpus scifact fiqa arguana; do
  cargo run --release --bin eval -- --data test-data/$ds --mode hybrid \
    --alpha 0.80 --compare-alpha 0.95
done
```

| Dataset | α=0.80 nDCG | α=0.95 nDCG | Δ | t | sig? |
|---------|-------------|-------------|---|---|------|
| NFCorpus (323q) | 0.3954 | 0.3962 | +0.0008 | +0.68 | no |
| SciFact (300q) | 0.7873 | 0.7875 | +0.0002 | +1.00 | no |
| FiQA (648q) | 0.4266 | 0.4335 | +0.0069 | +3.44 | **yes** |
| ArguAna (1406q) | 0.4263 | 0.4269 | +0.0006 | +1.39 | no |

**Conclusion**: 1/4 datasets significant (FiQA, t=3.44). FiQA is a financial Q&A corpus where
dense retrieval naturally dominates; the gain is dataset-specific. Deltas on the other three are
noise. **α=0.80 stays** — consistent midpoint, no regression risk.

---

## Experiment: Unified Qwen3.5 (ongoing)

**Hypothesis**: Replace both the reranker (0.6B) and expander (1.7B) with a single
Qwen3.5 model. Use DSPy MIPROv2 to optimize prompts offline against NFCorpus/SciFact,
then hardcode winning prompts in `src/llm/qwen.rs`.

### Model Comparison

| | Qwen3.5-0.8B | Qwen3.5-2B | Current combined |
|---|---|---|---|
| Params | 0.8B | 2B | 0.6B + 1.7B = 2.3B |
| GGUF (local) | Q8_0 812MB | Q4_K_M 1.3GB | ~1.6GB combined |
| Models to load | 1 | 1 | 2 |
| Architecture | Gated DeltaNet, 262K ctx | Gated DeltaNet, 262K ctx | Qwen3 transformer |

### Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1a: commit dirty tree | ✅ | 95b2ab1 |
| 1b: llama-cpp-2 → 0.1.137 | ✅ | Gated DeltaNet support |
| 1c: smoke tests | ✅ | both models load, generate, tokenize |
| 1c: functional tests | ✅ | expand + score_relevance pass |
| 2: DSPy prompt optimization | ⬜ | see below |
| 3: Rust integration | ✅ | `src/llm/qwen.rs` wired into pipeline |
| 4: benchmark runs | ⬜ | pending Phase 2 |

### Phase 2: DSPy Optimization

```bash
pip install dspy ollama
ollama pull qwen3.5:0.8b
ollama pull qwen3.5:2b

python research/export_eval_data.py        # exports NFCorpus/SciFact → artifacts/
python research/dspy_optimize.py           # MIPROv2 + BootstrapFewShot; saves artifacts/
```

Outputs: `research/artifacts/{model}_expander.json`, `{model}_reranker.json`, `{model}_prompts.txt`.
Paste winning prompts into `src/llm/qwen.rs` constants (marked `// ! DSPy-optimized prompt`).

### Benchmark Runs (planned)

| Run | Expander | Reranker | GGUF total | Target |
|-----|----------|----------|------------|--------|
| A (baseline) | qmd-1.7B | Qwen3-Reranker-0.6B | ~1.6GB | 0.4032 |
| B | Qwen3.5-0.8B | Qwen3.5-0.8B | ~812MB | ≥ 0.4032 |
| C | Qwen3.5-2B | Qwen3.5-2B | ~1.3GB | ≥ 0.4032 |
| D (ablation) | Qwen3.5-2B | Qwen3-Reranker-0.6B | ~1.9GB | — |
| E (ablation) | qmd-1.7B | Qwen3.5-2B | ~2.3GB | — |

```bash
# Run B
IR_QWEN_MODEL=~/local-models/Qwen3.5-0.8B-Q8_0.gguf \
  cargo run --release --bin eval -- --data test-data/nfcorpus --mode all

# Run C
IR_QWEN_MODEL=~/local-models/Qwen3.5-2B-Q4_K_M.gguf \
  cargo run --release --bin eval -- --data test-data/nfcorpus --mode all
```

### Decision Matrix

| Outcome | Action |
|---------|--------|
| 0.8B matches baseline nDCG | Ship 0.8B — 812MB for both roles |
| 2B matches, 0.8B doesn't | Ship 2B — still smaller than current 1.6GB |
| Neither matches | Keep current models; DSPy prompts still applicable |
| DSPy prompts improve fine-tuned models | Apply optimization to existing models too |

### Results

Benchmark runs B and C completed (Phase 2 / DSPy skipped — Rust integration tested directly).

#### NFCorpus (3,633 docs · 323 queries)

| Run | nDCG@10 | vs baseline | Notes |
|-----|---------|-------------|-------|
| A (baseline) | 0.4032 | — | qmd-1.7B + Qwen3-Reranker-0.6B |
| B (0.8B) | 0.3959 | −0.0073 (−1.8%) | Qwen3.5-0.8B-Q8_0, unified |
| C (2B) | 0.3956 | −0.0076 (−1.9%) | Qwen3.5-2B-Q4_K_M, unified |

#### SciFact (5,183 docs · 300 queries)

| Run | nDCG@10 | vs baseline | Notes |
|-----|---------|-------------|-------|
| A (baseline) | 0.7873 | — | |
| B (0.8B) | 0.7873 | 0 | identical — dataset near ceiling |
| C (2B) | 0.7873 | 0 | identical |

**Decision: keep current trio** (qmd-1.7B + Qwen3-Reranker-0.6B). Neither Qwen3.5 size
matches baseline on NFCorpus. SciFact is too easy to discriminate models (vector alone: 0.785).

Notable: 2B shows no improvement over 0.8B despite 2× size — reranking quality is not the
bottleneck; expansion quality or BM25 probe threshold matters more.

---

## Korean IR Benchmark (Ko-StrategyQA)

**Dataset**: Ko-StrategyQA — 9,251 Korean Wikipedia paragraphs · 592 test queries · binary relevance.
Multi-hop yes/no questions; each query requires finding 2–3 supporting paragraphs.
**Metric**: nDCG@10 (primary), Recall@10 (secondary).

```bash
scripts/bench-ko.sh bm25      # BM25 phase (no model, fast)
scripts/bench-ko.sh vector    # embed corpus once (~9k docs)
scripts/bench-ko.sh hybrid    # hybrid + rerank
scripts/bench-ko.sh --reset   # wipe all eval DBs
```

### Models

| Component | Model | Korean support |
|-----------|-------|---------------|
| Embedding | EmbeddingGemma 308M (768d) | 100+ languages — confirmed working |
| Reranker | Qwen3-Reranker-0.6B | 119 languages — confirmed working |
| Expander | qmd-expander-1.7B | Qwen3 base (119 langs), English SFT — **hurts Korean** (tested on MIRACL) |

### Preprocessors evaluated

| Preprocessor | Type | Dictionary | Runtime |
|---|---|---|---|
| none | unicode61 (FTS5 default) | — | — |
| kiwi | Neural POS tagger | Custom statistical | Python subprocess (~2s startup) |
| mecab | CRF tagger | mecab-ko-dic | Python subprocess (~0.3s startup) |
| lindera | CRF tagger | mecab-ko-dic (same) | Rust binary (~0s startup) |

### Results

| Mode | none | kiwi | mecab | lindera |
|------|------|------|-------|---------|
| bm25 | 0.0000 | 0.0053 | 0.0039 | 0.0039 |
| vector | 0.7992 | — | — | — |
| hybrid | 0.7992 | 0.7991 | 0.7984 | — |
| **hybrid+rerank** | 0.8138 | **0.8148** | 0.8137 | — |

Recall@10: vector=0.8674, hybrid+rerank(kiwi)=0.8756.

### Analysis

**BM25 is ineffective for this task.** Ko-StrategyQA multi-hop queries share almost no surface
terms with the supporting paragraphs. Unicode61 tokenizer scores 0.0000 — Korean agglutination
means "이스탄불의" (istanbul+possessive) and "이스탄불은" (istanbul+subject) are different FTS tokens
and never match. Morphological tokenizers recover some signal (kiwi: 0.0053) but BM25 remains
negligible compared to vector.

**EmbeddingGemma handles Korean extremely well.** Vector nDCG@10=0.7992, Recall@10=0.8674 with
no Korean-specific training — the model finds the correct supporting paragraph 87% of the time
in top-10. This confirms multilingual embedding capability is sufficient for Korean retrieval.

**Hybrid = vector** (0.7992 both). With α=0.80 and BM25 at 0.005, the BM25 component
contributes nothing to score fusion. Tokenizer choice is irrelevant for this dataset.

**Reranker adds +0.015 nDCG@10** (0.7992 → 0.8148). Qwen3-Reranker-0.6B correctly rescores
Korean query-document pairs despite English-heavy SFT. The reranker is the only component that
improves over pure vector on this task.

**Decision: lindera as the single ko preprocessor.** Kiwi's +0.001 nDCG in hybrid+rerank does
not justify its 2s Python startup per query. Mecab parity with lindera confirmed; lindera is
the production-safe choice (Rust, embedded dictionary, zero Python dep). See MIRACL and compound
benchmark below for the decompounding rationale.

---

## Korean IR Benchmark (MIRACL-Korean)

**Dataset**: MIRACL-Korean dev — 2,835 passages (547 relevant + 2,288 hard negatives) · 213 queries.
Factoid Wikipedia queries with direct term overlap — opposite of Ko-StrategyQA multi-hop.
Hard negatives sourced from BM25+DR retrieval (lexically similar but not relevant).
**Metric**: nDCG@10 (primary), Recall@10 (secondary).

```bash
uv run scripts/download-ko-miracl.py   # one-time setup
scripts/bench-ko-miracl.sh             # full run (BM25 parallel, model sequential)
scripts/bench-ko-miracl.sh bm25        # BM25 only
scripts/bench-ko-miracl.sh model       # model phases only
```

### Results

| Mode | none | kiwi | mecab | lindera |
|------|------|------|-------|---------|
| bm25 | 0.0009 | **0.1325** | 0.0460 | 0.0460 |
| **hybrid+rerank** | **0.8411** | 0.8429 | — | — |
| hybrid+expand+rerank | 0.8375 | — | — | — |

Recall@10: hybrid+rerank(none)=0.9699, hybrid+rerank(kiwi)=0.9699.

### Analysis

**BM25 is not fundamentally broken for Korean — Ko-StrategyQA was the outlier.** Unicode61
scores 0.0009 on MIRACL (vs 0.0000 on Ko-StrategyQA). Factoid queries share surface terms with
passages; multi-hop queries do not. The 0.0000 result was task-specific, not a language limit.

**Kiwi is 3× better than mecab/lindera on BM25 (0.1325 vs 0.0460).** The gap is tokenization
accuracy: kiwi's neural tagger correctly handles compound nouns and ambiguous morpheme boundaries
that mecab-ko-dic CRF gets wrong.

**Mecab and lindera are identical (0.0460 both)**, confirming they share the same underlying
dictionary and segmentation logic.

**Expander hurts Korean retrieval (0.8411 → 0.8375, −0.4% nDCG; Recall 0.9699 → 0.9633).**
`qmd-expander-1.7B` (English SFT) generates English or mixed-language sub-queries that dilute
the Korean vector signal. **Do not use the expander for Korean collections.**

**Hybrid+rerank is near-ceiling** (0.84 nDCG, 0.97 Recall@10). Kiwi adds +0.002 nDCG vs none
in hybrid+rerank — negligible in practice.

### Recommendation

- Disable expander for Korean collections.
- Reranker is the main lever (+0.015–0.027 nDCG). Always enable.
- **lindera is the ko preprocessor.** Kiwi's +0.002 nDCG in hybrid mode does not justify its
  2s Python startup per query. Lindera handles compound decompounding — see below.

---

## Korean Compound Decompounding Benchmark

**Hypothesis**: lindera `Mode::Decompose` (other_penalty_length_threshold=2) indexes compound
sub-parts, enabling BM25 to match queries whose terms only exist inside compound nouns.
Without decompounding, BM25 returns zero hits for these queries.

**Dataset**: 50 synthetic queries from ko-miracl corpus. Each query uses the sub-components
of a compound noun that does not appear decomposed anywhere in the raw corpus.

```bash
uv run scripts/gen-compound-bench.py   # generates test-data/ko-compound/
scripts/bench-compound.sh
```

### Results (ko-compound, nDCG@10)

| preprocessor | nDCG@10 | Recall@10 | note |
|---|---|---|---|
| none | 0.0000 | 0.0000 | compounds indexed whole — sub-parts missing from FTS |
| lindera | **0.6326** | **0.6400** | Mode::Decompose splits compounds, sub-parts indexed |

### Analysis

**None scores 0.0000 exactly** — confirming the query construction is correct. Sub-components
of selected compounds genuinely do not appear independently in the corpus (strict filter).

**Lindera scores 0.63 nDCG**, not 1.0 — expected: some compounds decompose further when used
as query terms (e.g. `협동조합` in the query decomposes to `협동 조합`, while the indexed
document has `협동조합체가` decomposed as `협동조합 체가`). Term mismatch at query time
causes partial recall.

**This confirms the core value of lindera over unicode61 for Korean BM25**: compound nouns
are common in Korean Wikipedia and would be completely missed without decompounding.

---

## Daemon mode

**Problem**: `ir search` cold-starts 3–7s per query due to model loading every invocation
(embedder 300M + expander 1.7B + reranker 0.6B = ~2.3B params, no cross-invocation caching).

**Solution**: `ir daemon start` — loads trio once with Metal, listens on Unix socket
(`~/.config/ir/daemon.sock`). `ir search` auto-detects and routes through daemon; falls back
to direct on connection failure.

```bash
ir daemon start      # foreground; models loaded once, Metal enabled
ir daemon status
ir daemon stop
ir search "query" -c kgeditor   # auto-routes through daemon if running
```

**DB handling**: daemon opens fresh WAL read-only connections per query (not `immutable=1`),
so live `ir index` / `ir embed` updates are visible immediately without restart.

**Model stack**: trio (nDCG@10=0.4032), Metal on by default (macOS). Override: `IR_GPU_LAYERS=0`.
