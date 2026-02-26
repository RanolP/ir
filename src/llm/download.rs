// Model auto-download via HuggingFace Hub.
// docs: https://docs.rs/hf-hub/latest/hf_hub/
//
// HF_HUB_OFFLINE=1  — skip network, use cache only (handled by hf-hub natively)
// IR_GPU_LAYERS=N   — override GPU layer count

use crate::error::{Error, Result};
use crate::llm::{find_model, hf_repos};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Return the local path for `filename`, downloading from HF if not found locally.
pub fn ensure_model(filename: &str) -> Result<PathBuf> {
    if let Some(p) = find_model(filename) {
        return Ok(p);
    }

    let (repo_id, hf_filename) = hf_repos::for_filename(filename).ok_or_else(|| {
        Error::Other(format!(
            "model '{filename}' not found and no HF repo configured.\n\
             Set the appropriate IR_*_MODEL env var or add a directory to IR_MODEL_DIRS."
        ))
    })?;

    // If a case-variant file already exists (e.g. q4_k_m vs Q4_K_M), reuse it
    // and place a stable alias at ~/.cache/ir/models/{filename}.
    if hf_filename != filename {
        if let Some(existing) = find_model(hf_filename) {
            return Ok(with_local_alias(filename, existing));
        }
    }

    eprintln!("downloading {filename} from {repo_id}/{hf_filename}...");

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| Error::Other(format!("hf-hub init: {e}")))?;

    let downloaded = api
        .model(repo_id.to_string())
        .get(hf_filename)
        .map_err(|e| Error::Other(format!("download {hf_filename} from {repo_id}: {e}")))?;

    Ok(with_local_alias(filename, downloaded))
}

fn with_local_alias(filename: &str, source: PathBuf) -> PathBuf {
    match ensure_local_alias(filename, &source) {
        Ok(Some(alias)) => alias,
        Ok(None) => source,
        Err(err) => {
            eprintln!("warning: {err}");
            source
        }
    }
}

fn ensure_local_alias(filename: &str, source: &Path) -> Result<Option<PathBuf>> {
    let Some(cache_dir) = dirs::cache_dir() else {
        return Ok(None);
    };

    let alias = cache_dir.join("ir").join("models").join(filename);
    let Some(parent) = alias.parent() else {
        return Err(Error::Other(format!(
            "invalid model alias path: {}",
            alias.display()
        )));
    };

    std::fs::create_dir_all(parent).map_err(|e| {
        Error::Other(format!(
            "create local model cache dir '{}': {e}",
            parent.display()
        ))
    })?;

    match std::fs::symlink_metadata(&alias) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                let existing_target = std::fs::read_link(&alias).map_err(|e| {
                    Error::Other(format!(
                        "read existing model alias '{}': {e}",
                        alias.display()
                    ))
                })?;
                if existing_target == source {
                    return Ok(Some(alias));
                }
                std::fs::remove_file(&alias).map_err(|e| {
                    Error::Other(format!(
                        "remove stale model alias '{}' -> '{}': {e}",
                        alias.display(),
                        existing_target.display()
                    ))
                })?;
            } else if meta.file_type().is_file() {
                // Respect an existing real file in the canonical location.
                return Ok(Some(alias));
            } else {
                return Err(Error::Other(format!(
                    "model alias path exists but is not a file: {}",
                    alias.display()
                )));
            }
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => {
            return Err(Error::Other(format!(
                "inspect model alias path '{}': {e}",
                alias.display()
            )));
        }
    }

    create_link(source, &alias).map_err(|e| {
        Error::Other(format!(
            "create model alias '{}' -> '{}': {e}",
            alias.display(),
            source.display()
        ))
    })?;

    Ok(Some(alias))
}

#[cfg(unix)]
fn create_link(source: &Path, alias: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, alias)
}

#[cfg(windows)]
fn create_link(source: &Path, alias: &Path) -> std::io::Result<()> {
    // Fallback to hard links when symlink permissions are restricted.
    std::os::windows::fs::symlink_file(source, alias).or_else(|_| std::fs::hard_link(source, alias))
}

#[cfg(not(any(unix, windows)))]
fn create_link(source: &Path, alias: &Path) -> std::io::Result<()> {
    std::fs::hard_link(source, alias)
}

#[cfg(test)]
mod tests {
    use crate::llm::{hf_repos, models};

    #[test]
    fn for_filename_resolves_all_known_models() {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(false)
            .build()
            .expect("hf-hub init");

        for filename in &[models::EMBEDDING, models::RERANKER, models::EXPANDER] {
            let (repo_id, hf_filename) = hf_repos::for_filename(filename)
                .unwrap_or_else(|| panic!("no mapping for {filename}"));
            let url = api.model(repo_id.to_string()).url(hf_filename);
            assert!(
                url.contains(repo_id),
                "url for {filename} missing repo_id: {url}"
            );
        }
    }

    #[test]
    #[ignore] // live network — run with: cargo test download -- --ignored
    fn embedding_repo_is_reachable() {
        let (repo_id, _) = hf_repos::EMBEDDING;
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(false)
            .build()
            .expect("hf-hub init");
        let info = api.model(repo_id.to_string()).info().expect("HF repo info");
        assert!(!info.siblings.is_empty(), "repo has no files?");
    }

    #[test]
    fn unknown_filename_returns_error() {
        // Ensure find_model returns None for a name with no HF mapping
        let result = hf_repos::for_filename("no-such-model.gguf");
        assert!(result.is_none());
    }
}
