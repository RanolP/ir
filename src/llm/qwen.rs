// Unified Qwen3.5 model: handles both reranking and query expansion in one GGUF load.
// docs: https://docs.rs/llama-cpp-2/latest/llama_cpp_2/
//
// Reranker role:  Yes/No logit scoring (same protocol as reranker.rs)
// Expander role:  autoregressive generation → lex:/vec:/hyde: lines
//
// Prompts are DSPy-MIPROv2 optimized (see research/dspy_optimize.py).
// Run research/dspy_optimize.py to re-optimize; hardcode results here.
//
// env: IR_QWEN_MODEL — full path or directory containing the GGUF

use crate::error::{Error, Result};
use crate::llm::{LlamaBackend, model_load_params, models};
use crate::llm::expander::{SubQuery, fallback, parse_output};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use std::num::NonZeroU32;
use std::path::Path;

const RERANK_CONTEXT_SIZE: u32 = 2048;
const EXPAND_CONTEXT_SIZE: u32 = 2048;
const MAX_DOC_CHARS: usize = 6000;
const MAX_EXPAND_TOKENS: usize = 300;

pub struct Qwen35 {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    yes_token_id: i32,
    no_token_id: i32,
}

impl Qwen35 {
    pub fn load(path: &Path) -> Result<Self> {
        let backend = crate::llm::init_backend()?;
        let model = LlamaModel::load_from_file(backend, path, &model_load_params())
            .map_err(|e| Error::Other(format!("load qwen3.5: {e}")))?;

        let yes_tokens = model
            .str_to_token("Yes", AddBos::Never)
            .map_err(|e| Error::Other(format!("tokenize 'Yes': {e}")))?;
        let no_tokens = model
            .str_to_token("No", AddBos::Never)
            .map_err(|e| Error::Other(format!("tokenize 'No': {e}")))?;

        // Use last token of each (handles BPE subword splits)
        let yes_id = yes_tokens.last().map(|t| t.0).unwrap_or(0);
        let no_id = no_tokens.last().map(|t| t.0).unwrap_or(1);

        Ok(Self {
            backend,
            model,
            yes_token_id: yes_id,
            no_token_id: no_id,
        })
    }

    pub fn load_default() -> Result<Self> {
        // Try IR_QWEN_MODEL env first, then fall back to default filename search.
        if let Some(path) = resolve_qwen_env() {
            return Self::load(&path);
        }
        // Prefer 2B, fall back to 0.8B
        for filename in &[models::QWEN35_2B, models::QWEN35_0_8B] {
            if let Some(path) = crate::llm::find_model(filename) {
                return Self::load(&path);
            }
        }
        Err(Error::Other(format!(
            "Qwen3.5 model not found. Set IR_QWEN_MODEL or place {} in ~/local-models/",
            models::QWEN35_2B
        )))
    }

    /// Score relevance of a document to a query. Returns P(Yes) in [0, 1].
    ///
    /// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
    pub fn score_relevance(&self, query: &str, doc: &str) -> Result<f64> {
        let doc_truncated = if doc.len() > MAX_DOC_CHARS {
            &doc[..doc.floor_char_boundary(MAX_DOC_CHARS)]
        } else {
            doc
        };

        let prompt = format!(
            "<|im_start|>system\n\
             Judge whether the Document meets the requirements based on the Query and the Instruct provided. \
             Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n\
             <|im_start|>user\n\
             <Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\
             <Query>: {query}\n\
             <Document>: {doc_truncated}<|im_end|>\n\
             <|im_start|>assistant\n\
             <think>\n\
             </think>\n"
        );

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(RERANK_CONTEXT_SIZE))
            .with_offload_kqv(false)
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);

        let mut ctx = self
            .model
            .new_context(self.backend, ctx_params)
            .map_err(|e| Error::Other(format!("rerank context: {e}")))?;

        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never) // ! ChatML starts with <|im_start|>; extra BOS confuses model
            .map_err(|e| Error::Other(format!("tokenize: {e}")))?;

        if tokens.is_empty() {
            return Ok(0.0);
        }

        let n = tokens.len().min(RERANK_CONTEXT_SIZE as usize - 1);
        let mut batch = LlamaBatch::new(n, 1);
        for (i, &tok) in tokens[..n].iter().enumerate() {
            batch
                .add(tok, i as i32, &[0], i == n - 1)
                .map_err(|e| Error::Other(format!("batch add: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| Error::Other(format!("decode: {e}")))?;

        let logits = ctx.get_logits_ith((n - 1) as i32);
        let yes_idx = self.yes_token_id as usize;
        let no_idx = self.no_token_id as usize;
        if yes_idx >= logits.len() || no_idx >= logits.len() {
            return Err(Error::Other(format!(
                "token id out of range: yes={yes_idx}, no={no_idx}, vocab={}",
                logits.len()
            )));
        }

        let yes_logit = logits[yes_idx];
        let no_logit = logits[no_idx];
        let max_logit = yes_logit.max(no_logit);
        let yes_exp = (yes_logit - max_logit).exp() as f64;
        let no_exp = (no_logit - max_logit).exp() as f64;

        Ok(yes_exp / (yes_exp + no_exp))
    }

    /// Expand a query into typed sub-queries (lex/vec/hyde). Falls back on parse failure.
    ///
    /// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
    pub fn expand(&self, query: &str) -> Result<Vec<SubQuery>> {
        let prompt = build_expand_prompt(query);
        let raw = self.generate_expand(&prompt)?;
        let parsed = parse_output(&raw);

        let query_lower = query.to_lowercase();
        let valid = parsed.iter().any(|s| {
            s.text
                .split_whitespace()
                .any(|w| query_lower.contains(&w.to_lowercase()))
        });

        if parsed.is_empty() || !valid {
            Ok(fallback(query))
        } else {
            Ok(parsed)
        }
    }

    fn generate_expand(&self, prompt: &str) -> Result<String> {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(EXPAND_CONTEXT_SIZE))
            .with_offload_kqv(false)
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);

        let mut ctx = self
            .model
            .new_context(self.backend, ctx_params)
            .map_err(|e| Error::Other(format!("expand context: {e}")))?;

        let prompt_tokens = self
            .model
            .str_to_token(prompt, AddBos::Never) // ! ChatML prompt; no extra BOS
            .map_err(|e| Error::Other(format!("tokenize expand: {e}")))?;

        let n_prompt = prompt_tokens.len();
        if n_prompt == 0 {
            return Ok(String::new());
        }

        let mut batch = LlamaBatch::new(n_prompt, 1);
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            batch
                .add(tok, i as i32, &[0], i == n_prompt - 1)
                .map_err(|e| Error::Other(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Error::Other(format!("decode prompt: {e}")))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::top_k(20),
            LlamaSampler::top_p(0.8, 1),
            LlamaSampler::dist(42),
        ]);

        let mut output = String::new();
        let mut n_cur = n_prompt as i32;

        for _ in 0..MAX_EXPAND_TOKENS {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let bytes = self
                .model
                .token_to_piece_bytes(token, 32, true, None)
                .map_err(|e| Error::Other(format!("token_to_piece_bytes: {e}")))?;
            output.push_str(&String::from_utf8_lossy(&bytes));

            let mut next = LlamaBatch::new(1, 1);
            next.add(token, n_cur, &[0], true)
                .map_err(|e| Error::Other(format!("batch next: {e}")))?;
            ctx.decode(&mut next)
                .map_err(|e| Error::Other(format!("decode next: {e}")))?;
            n_cur += 1;
        }

        Ok(output)
    }
}

/// // ! DSPy-optimized prompt — do not edit manually; re-run research/dspy_optimize.py
fn build_expand_prompt(query: &str) -> String {
    format!(
        "<|im_start|>system\n\
         Generate search sub-queries for document retrieval. \
         Output exactly three lines: lex (2-5 keywords for BM25), \
         vec (natural language reformulation), hyde (1-2 sentence hypothetical answer passage).<|im_end|>\n\
         <|im_start|>user\n\
         Query: {query}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn resolve_qwen_env() -> Option<std::path::PathBuf> {
    let raw = std::env::var_os("IR_QWEN_MODEL")?;
    let path = std::path::PathBuf::from(raw);
    if path.is_file() {
        return Some(path);
    }
    // Directory: try both model filenames
    if path.is_dir() {
        for filename in &[models::QWEN35_2B, models::QWEN35_0_8B] {
            let candidate = path.join(filename);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn load_0_8b_and_tokenize() {
        // Phase 1c smoke test: verify Gated DeltaNet works in llama-cpp-2
        // Download: huggingface.co/unsloth/Qwen3.5-0.8B-GGUF → ~/local-models/
        let path = dirs::home_dir()
            .unwrap()
            .join("local-models")
            .join(models::QWEN35_0_8B);
        assert!(path.exists(), "model not found: {}", path.display());

        let q = Qwen35::load(&path).expect("load 0.8B");

        // Smoke: tokenize a short string
        let tokens = q
            .model
            .str_to_token("hello world", AddBos::Never)
            .expect("tokenize");
        assert!(!tokens.is_empty(), "tokenization returned empty");

        // Generate 10 tokens from a trivial prompt
        let result = q.generate_expand("hello");
        assert!(result.is_ok(), "generate failed: {:?}", result.err());
        println!("0.8B output: {:?}", result.unwrap());
    }

    #[test]
    #[ignore]
    fn load_2b_and_tokenize() {
        let path = dirs::home_dir()
            .unwrap()
            .join("local-models")
            .join(models::QWEN35_2B);
        assert!(path.exists(), "model not found: {}", path.display());

        let q = Qwen35::load(&path).expect("load 2B");
        let tokens = q
            .model
            .str_to_token("hello world", AddBos::Never)
            .expect("tokenize");
        assert!(!tokens.is_empty());
        println!("2B vocab size: {}", q.model.n_vocab());
    }

    #[test]
    #[ignore]
    fn expand_returns_valid_subqueries() {
        use crate::llm::expander::SubQueryKind;
        let q = Qwen35::load_default().expect("load model");
        let subs = q.expand("rust memory management").expect("expand");
        assert!(!subs.is_empty());
        let any_relevant = subs
            .iter()
            .any(|s| s.text.contains("rust") || s.text.contains("memory"));
        assert!(any_relevant, "no relevant sub-query in: {subs:?}");

        let kinds: Vec<SubQueryKind> = subs.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&SubQueryKind::Lex));
        assert!(kinds.contains(&SubQueryKind::Vec));
    }

    #[test]
    #[ignore]
    fn score_relevance_orders_correctly() {
        let q = Qwen35::load_default().expect("load model");
        let relevant = q
            .score_relevance(
                "rust memory management",
                "Rust uses ownership and borrowing to manage memory without a garbage collector",
            )
            .expect("score");
        let irrelevant = q
            .score_relevance(
                "rust memory management",
                "Python uses a garbage collector. JavaScript also has automatic memory management.",
            )
            .expect("score");
        assert!(
            relevant > irrelevant,
            "relevant={relevant:.3} should > irrelevant={irrelevant:.3}"
        );
    }
}
