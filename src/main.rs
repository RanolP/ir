mod cli;
mod config;
mod daemon;
mod db;
mod error;
mod index;
mod llm;
mod preprocess;
mod search;
mod types;

use clap::Parser;
use cli::{Cli, CollectionCmd, Command, DaemonCmd, PreprocessorCmd, output};
use config::{Config, collection_db_path};
use error::Result;
use types::{Collection, SearchMode};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Collection { cmd } => handle_collection(cmd),
        Command::Status => handle_status(),
        Command::Update { collection, force } => handle_update(collection, force),
        Command::Embed { collection, force } => handle_embed(collection, force),
        Command::Search {
            query,
            mode,
            limit,
            min_score,
            collections,
            all,
            full,
            json,
            csv,
            md,
            files,
            verbose,
        } => handle_search(
            query.join(" "),
            mode,
            if all { crate::db::vectors::KNN_MAX } else { limit },
            min_score,
            collections,
            full,
            json,
            csv,
            md,
            files,
            verbose,
        ),
        Command::Get { .. } => {
            eprintln!("not yet implemented");
            Ok(())
        }
        Command::Daemon { cmd } => match cmd {
            DaemonCmd::Start { timeout } => daemon::start_server(timeout),
            DaemonCmd::Stop => daemon::stop(),
            DaemonCmd::Status => daemon::status(),
        },
        Command::Preprocessor { cmd } => handle_preprocessor(cmd),
    }
}

fn handle_collection(cmd: CollectionCmd) -> Result<()> {
    let mut config = Config::load()?;
    match cmd {
        CollectionCmd::Add {
            name,
            path,
            glob,
            exclude,
            description,
            preprocessor,
        } => {
            // Validate aliases before mutating config.
            for alias in &preprocessor {
                if !config.preprocessors.contains_key(alias.as_str()) {
                    return Err(error::Error::Other(format!(
                        "preprocessor alias '{alias}' not registered. Run: ir preprocessor add {alias} <command>"
                    )));
                }
            }
            let resolved = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone().into());
            config.add_collection(Collection {
                name: name.clone(),
                path: resolved.to_string_lossy().into_owned(),
                globs: glob,
                excludes: exclude,
                description,
                preprocessor: if preprocessor.is_empty() { None } else { Some(preprocessor) },
            })?;
            config.save()?;
            println!("added collection '{name}'");
        }
        CollectionCmd::Rm { name, purge } => {
            config.remove_collection(&name)?;
            config.save()?;
            if purge {
                let db_path = collection_db_path(&name);
                if db_path.exists() {
                    std::fs::remove_file(&db_path)?;
                    println!("removed collection '{name}' and deleted database");
                } else {
                    println!("removed collection '{name}'");
                }
            } else {
                println!("removed collection '{name}' (database kept)");
            }
        }
        CollectionCmd::Rename { old, new } => {
            config.rename_collection(&old, &new)?;
            config.save()?;
            println!("renamed '{old}' → '{new}'");
        }
        CollectionCmd::SetPath { name, path } => {
            config.set_collection_path(&name, &path)?;
            config.save()?;
            println!("updated path for '{name}' → {path}");
            println!("run `ir daemon stop` then `ir update {name}` to sync");
        }
        CollectionCmd::Ls => {
            if config.collections.is_empty() {
                println!("no collections configured");
            } else {
                for c in &config.collections {
                    if let Some(desc) = &c.description {
                        println!("{:<20} {}  # {}", c.name, c.path, desc);
                    } else {
                        println!("{:<20} {}", c.name, c.path);
                    }
                }
            }
        }
    }
    Ok(())
}

fn handle_status() -> Result<()> {
    let config = Config::load()?;
    println!("collections: {}", config.collections.len());
    for col in &config.collections {
        let db_path = collection_db_path(&col.name);
        let db_exists = db_path.exists();
        let status = if db_exists { "indexed" } else { "not indexed" };
        let size = if db_exists {
            let bytes = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
            format!("{:.1} MB", bytes as f64 / 1_048_576.0)
        } else {
            String::new()
        };
        println!("  {:<20} {:<12} {}  {}", col.name, status, col.path, size);
    }
    Ok(())
}

fn handle_update(collection: Option<String>, force: bool) -> Result<()> {
    let config = Config::load()?;
    let cols: Vec<_> = match &collection {
        Some(name) => {
            let c = config
                .get_collection(name)
                .ok_or_else(|| error::Error::CollectionNotFound(name.clone()))?;
            vec![c]
        }
        None => config.collections.iter().collect(),
    };

    for col in cols {
        let db_path = collection_db_path(&col.name);
        let db = db::CollectionDb::open(&col.name, &db_path)?;
        println!("updating '{}'…", col.name);
        let opts = index::UpdateOptions { force };
        let (added, updated, deactivated) = index::update(&db, col, &opts)?;
        println!(
            "  {} added, {} updated, {} deactivated",
            added, updated, deactivated
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_search(
    query: String,
    mode: String,
    limit: usize,
    min_score: Option<f64>,
    collection_filter: Vec<String>,
    full: bool,
    json: bool,
    csv: bool,
    md: bool,
    files: bool,
    verbose: bool,
) -> Result<()> {
    let fmt = if json {
        output::Format::Json
    } else if csv {
        output::Format::Csv
    } else if md {
        output::Format::Markdown
    } else if files {
        output::Format::Files
    } else {
        output::Format::Pretty
    };

    let config = Config::load()?;
    let collection_names = resolve_collections(&config, &collection_filter)?;
    let search_mode: SearchMode = mode.parse().map_err(error::Error::Other)?;

    // Open DBs for in-process BM25 (tier-0; also used as fallback).
    let cols: Vec<_> = collection_names.iter()
        .filter_map(|name| config.get_collection(name))
        .collect();
    let dbs: Vec<db::CollectionDb> = cols.iter()
        .map(|c| {
            db::CollectionDb::open_rw(&c.name, &collection_db_path(&c.name))
        })
        .collect::<Result<Vec<_>>>()?;

    // Tier-0: BM25 in-process, no model needed.
    let bm25_req = search::fan_out::SearchRequest { query: &query, limit, min_score };
    let bm25_results = search::fan_out::bm25(&dbs, &bm25_req)?;

    // Mode dispatch before going to daemon.
    match search_mode {
        // bm25 mode: return BM25 results directly, no daemon needed.
        SearchMode::Bm25 => {
            output::print_results(&bm25_results, fmt, full);
            return Ok(());
        }
        // vector mode: skip BM25 shortcut — go straight to daemon.
        SearchMode::Vector => {}
        // hybrid mode: strong BM25 signal shortcuts LLM work.
        SearchMode::Hybrid => {
            if search::hybrid::is_bm25_strong_signal(&bm25_results) {
                if !daemon::is_running() { let _ = daemon::start_in_background(); }
                output::print_results(&bm25_results, fmt, full);
                return Ok(());
            }
        }
    }

    // Need better results — ensure daemon is running.
    if !daemon::is_running() {
        if let Err(e) = daemon::start_in_background() {
            eprintln!("note: could not start daemon ({e})");
            output::print_results(&bm25_results, fmt, full);
            return Ok(());
        }
    }

    let req = daemon::DaemonRequest {
        query: query.clone(),
        collections: collection_names.clone(),
        limit,
        min_score,
        mode: mode.clone(),
        verbose,
    };

    eprint!("searching...");
    if !daemon::wait_ready(3_000) {
        eprintln!();
        output::print_results(&bm25_results, fmt, full);
        return Ok(());
    }

    // If tier-2 was ready before connecting, this query gets full hybrid.
    let tier2_before = daemon::is_tier2_ready();

    let tier1 = match daemon::query(&req) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("\nnote: daemon query error: {e}");
            output::print_results(&bm25_results, fmt, full);
            return Ok(());
        }
    };

    // Daemon already had full hybrid, or non-hybrid mode — tier-1 result is final.
    if tier2_before || search_mode != SearchMode::Hybrid {
        eprintln!();
        for line in &tier1.log { eprintln!("{line}"); }
        output::print_results(&to_search_results(tier1.results), fmt, full);
        return Ok(());
    }

    // Check if tier-1 result is strong enough to skip tier-2.
    let top = tier1.results.first().map(|r| r.score).unwrap_or(0.0);
    let gap = tier1.results.get(1).map(|r| top - r.score).unwrap_or(top);
    if top >= search::hybrid::STRONG_SIGNAL_FLOOR
        && top * gap >= search::hybrid::STRONG_SIGNAL_PRODUCT
    {
        eprintln!();
        for line in &tier1.log { eprintln!("{line}"); }
        output::print_results(&to_search_results(tier1.results), fmt, full);
        return Ok(());
    }

    // Tier-2: wait for expander+reranker, then re-query for full hybrid.
    eprint!(" enhancing...");
    if !daemon::wait_tier2(7_000) {
        eprintln!();
        for line in &tier1.log { eprintln!("{line}"); }
        output::print_results(&to_search_results(tier1.results), fmt, full);
        return Ok(());
    }

    match daemon::query(&req) {
        Ok(tier2) => {
            eprintln!();
            for line in &tier2.log { eprintln!("{line}"); }
            output::print_results(&to_search_results(tier2.results), fmt, full);
        }
        Err(_) => {
            eprintln!();
            for line in &tier1.log { eprintln!("{line}"); }
            output::print_results(&to_search_results(tier1.results), fmt, full);
        }
    }
    Ok(())
}

fn to_search_results(daemon_results: Vec<daemon::DaemonResult>) -> Vec<types::SearchResult> {
    daemon_results.into_iter()
        .map(|r| types::SearchResult {
            collection: r.collection,
            path: r.path,
            title: r.title,
            score: r.score,
            snippet: if r.snippet.is_empty() { None } else { Some(r.snippet) },
            hash: r.hash,
            doc_id: r.doc_id,
        })
        .collect()
}

fn resolve_collections(config: &Config, filter: &[String]) -> Result<Vec<String>> {
    if filter.is_empty() {
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(col) = config::detect_collection(&config.collections, &cwd) {
            Ok(vec![col.name.clone()])
        } else {
            Ok(config.collections.iter().map(|c| c.name.clone()).collect())
        }
    } else {
        let unknown: Vec<&str> = filter
            .iter()
            .filter(|name| config.get_collection(name).is_none())
            .map(|s| s.as_str())
            .collect();
        if !unknown.is_empty() {
            return Err(error::Error::Other(format!(
                "unknown collection(s): {}",
                unknown.join(", ")
            )));
        }
        Ok(filter.to_vec())
    }
}

fn handle_preprocessor(cmd: PreprocessorCmd) -> Result<()> {
    let mut config = Config::load()?;
    match cmd {
        PreprocessorCmd::Add { alias, command } => {
            if command.is_empty() {
                return Err(error::Error::Other("command must not be empty".into()));
            }
            let cmd_str = command.join(" ");
            config.add_preprocessor(&alias, &cmd_str)?;
            config.save()?;
            println!("registered preprocessor '{alias}': {cmd_str}");
        }
        PreprocessorCmd::Install { lang } => {
            install_preprocessor(&mut config, &lang)?;
        }
        PreprocessorCmd::List => {
            if config.preprocessors.is_empty() {
                println!("no preprocessors registered");
            } else {
                let mut entries: Vec<_> = config.preprocessors.iter().collect();
                entries.sort_by_key(|(k, _)| k.as_str());
                for (alias, cmd) in entries {
                    println!("{:<12} {}", alias, cmd);
                }
            }
        }
        PreprocessorCmd::Remove { alias } => {
            config.remove_preprocessor(&alias)?;
            config.save()?;
            println!("removed preprocessor '{alias}'");
        }
    }
    Ok(())
}

/// Download/install a bundled preprocessor and register it.
fn install_preprocessor(config: &mut Config, lang: &str) -> Result<()> {
    enum Kind { Script { repo_subdir: &'static str, script_name: &'static str }, Cargo { crate_name: &'static str } }
    struct Entry { alias: &'static str, kind: Kind }

    let known: &[Entry] = &[
        Entry { alias: "ko-kiwi",    kind: Kind::Script { repo_subdir: "ko", script_name: "kiwi-tokenize" } },
        Entry { alias: "ko-mecab",   kind: Kind::Script { repo_subdir: "ko", script_name: "mecab-tokenize" } },
        Entry { alias: "ko-lindera", kind: Kind::Cargo  { crate_name: "lindera-tokenize" } },
        Entry { alias: "ja",         kind: Kind::Script { repo_subdir: "ja", script_name: "mecab-tokenize" } },
    ];

    let entry = known
        .iter()
        .find(|e| e.alias == lang)
        .ok_or_else(|| error::Error::Other(
            format!("unknown lang '{lang}'. Available: ko-kiwi, ko-mecab, ko-lindera, ja")
        ))?;

    let cmd_str = match &entry.kind {
        Kind::Script { repo_subdir, script_name } => {
            let install_dir = config::ir_dir().join("preprocessors").join(repo_subdir);
            std::fs::create_dir_all(&install_dir)?;
            let script_path = install_dir.join(script_name);
            let url = format!(
                "https://raw.githubusercontent.com/vlwkaos/ir/korean/preprocessors/{repo_subdir}/{script_name}"
            );
            let status = std::process::Command::new("curl")
                .args(["-fsSL", &url, "-o", &script_path.to_string_lossy()])
                .status()
                .map_err(|e| error::Error::Other(format!("curl: {e}")))?;
            if !status.success() {
                return Err(error::Error::Other(format!(
                    "download failed. Install manually to {}", script_path.display()
                )));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755))
                    .map_err(error::Error::Io)?;
            }
            script_path.to_string_lossy().into_owned()
        }
        Kind::Cargo { crate_name } => {
            // Installs binary to ~/.cargo/bin/<crate_name>.
            // Requires: cargo install <crate_name> (published on crates.io).
            let status = std::process::Command::new("cargo")
                .args(["install", crate_name])
                .status()
                .map_err(|e| error::Error::Other(format!("cargo: {e}")))?;
            if !status.success() {
                return Err(error::Error::Other(format!(
                    "cargo install {crate_name} failed. Build manually:\n  cd preprocessors/ko/lindera-tokenize && cargo build --release\n  ir preprocessor add ko-lindera ./target/release/{crate_name}"
                )));
            }
            // Binary lands in ~/.cargo/bin/ which is on PATH; register by name.
            crate_name.to_string()
        }
    };

    let alias = entry.alias;
    config.add_preprocessor(alias, &cmd_str)?;
    config.save()?;
    println!("installed '{alias}' preprocessor → {cmd_str}");
    Ok(())
}

fn handle_embed(collection: Option<String>, force: bool) -> Result<()> {
    let config = Config::load()?;
    let cols: Vec<_> = match &collection {
        Some(name) => {
            let c = config
                .get_collection(name)
                .ok_or_else(|| error::Error::CollectionNotFound(name.clone()))?;
            vec![c]
        }
        None => config.collections.iter().collect(),
    };

    println!("loading embedding model…");
    let embedder = llm::embedding::Embedder::load_default()?;

    for col in cols {
        let db_path = collection_db_path(&col.name);
        let db = db::CollectionDb::open(&col.name, &db_path)?;
        println!("embedding '{}'…", col.name);
        let opts = index::embed::EmbedOptions { force };
        let (docs, chunks) = index::embed::embed(&db, &embedder, &opts, llm::models::EMBEDDING)?;
        println!("  {} documents, {} chunks embedded", docs, chunks);
    }
    Ok(())
}
