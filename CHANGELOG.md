## [0.5.0] - 2026-03-10

### Features

- `ir collection set-path`: update the root path of an existing collection without re-creating it ([`1ee1330`](https://github.com/vlwkaos/ir/commit/1ee1330))
- Eval: paired t-test across pipeline modes; per-dataset label in summary table ([`d7db622`](https://github.com/vlwkaos/ir/commit/d7db622))

### Refactor

- Progressive search pipeline: BM25+vector always fused; expansion gated on expander presence; strong-signal shortcut exits before expansion ([`2e987f0`](https://github.com/vlwkaos/ir/commit/2e987f0))

### Benchmark (4 BEIR datasets, nDCG@10)

EmbeddingGemma 300M + qmd-expander-1.7B + Qwen3-Reranker-0.6B:

| Dataset | BM25 | Vector | Hybrid | +Reranker |
|---|---|---|---|---|
| NFCorpus (323q) | 0.2046 | 0.3898 | 0.3954 | **0.4001** |
| SciFact (300q) | 0.0500 | 0.7847 | 0.7873 | **0.7797** |
| FiQA (648q) | 0.0298 | 0.4324 | 0.4266 | **0.4567** |
| ArguAna (1406q) | 0.0012 | 0.4264 | 0.4263 | **0.4879** |

BM25 fusion provides no statistically significant lift on any dataset (paired t-test). Reranker adds up to +14.5% on conversational/argument retrieval.

### Other

- Crate published as `ir-search` on crates.io (`ir` name was taken) ([`39dfefd`](https://github.com/vlwkaos/ir/commit/39dfefd))
- BEIR dataset download script (`scripts/download-beir.sh`) ([`ff71879`](https://github.com/vlwkaos/ir/commit/ff71879))

## [0.4.0] - 2026-03-06

### Features

- Daemon mode: keeps models warm between queries, auto-starts on first search ([`a9d1996`](https://github.com/vlwkaos/ir/commit/a9d1996))
- Global expander output cache (`~/.config/ir/expander_cache.sqlite`): repeated queries skip LLM inference entirely ([`407ea55`](https://github.com/vlwkaos/ir/commit/407ea55))
- Reranker score cache now includes model_id in key, preventing cross-model cache collisions ([`8b6c82e`](https://github.com/vlwkaos/ir/commit/8b6c82e))
- Eval harness: cache query embeddings and per-query results across runs ([`fe51038`](https://github.com/vlwkaos/ir/commit/fe51038))

### Performance (macOS M4 Max, vs qmd, same models and query)

| | ir | qmd | ratio |
|---|---:|---:|---|
| Cold (no cache) | 3.0s | 9.5s | 3× faster |
| Warm (daemon + caches hot) | 30ms | 840ms | 28× faster |

Cold: expander ~2.9s + reranker inference (ir caps at 20 candidates, qmd at 40).
Warm: both expander and reranker cache-hit; qmd pays ~800ms process spawn per call.

### Refactor

- Extract `scoring.rs` (Scorer trait + batch inference) and `generate.rs` (autoregressive generation) from reranker ([`8b6c82e`](https://github.com/vlwkaos/ir/commit/8b6c82e))

### Other

- DSPy experiment: BootstrapFewShot + ChatAdapter; Qwen3.5 underperforms trio by ~1.9% nDCG@10 ([`6f268fc`](https://github.com/vlwkaos/ir/commit/6f268fc))
- bench.sh script for sequential multi-config eval runs ([`55b61f5`](https://github.com/vlwkaos/ir/commit/55b61f5))

## [0.3.1-pre] - 2026-03-03

### Bug Fixes

- Auto CPU fallback when Metal context creation fails in sandboxed environments ([`299780a`](https://github.com/vlwkaos/ir/commit/299780a))
- Open search connections immutable/read-only to avoid WAL shm writes in sandbox ([`aa307c2`](https://github.com/vlwkaos/ir/commit/aa307c2))

### Other

- Move config and data to XDG-style `~/.config/ir` (cross-platform, sandbox-accessible) ([`295b3bb`](https://github.com/vlwkaos/ir/commit/295b3bb))
- DSPy optimizer: add structured logging, `--resume` flag, ollama smoke-test ([`2fdd0e9`](https://github.com/vlwkaos/ir/commit/2fdd0e9))
