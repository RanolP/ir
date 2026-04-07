use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub globs: Vec<String>,
    #[serde(default)]
    pub excludes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Preprocessor aliases to apply (in order) before FTS indexing and query time.
    /// Each alias must be registered in config.preprocessors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preprocessor: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub collection: String,
    pub path: String,
    pub title: String,
    pub score: f64,
    pub snippet: Option<String>,
    pub hash: String,
    pub doc_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl SearchResult {
    pub fn sort_desc(results: &mut [Self]) {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Controls stderr output from `search_core`.
/// - `Quiet`: no stderr at all (MCP stdio transport)
/// - `Normal`: progress indicators + daemon decision logs
/// - `Verbose`: Normal + daemon timing lines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Verbosity {
    Quiet,
    #[default]
    Normal,
    Verbose,
}

impl Verbosity {
    /// True when progress indicators ("searching...", "enhancing...") should print.
    pub fn show_progress(self) -> bool {
        self != Self::Quiet
    }
    /// True when daemon log lines (decisions, errors) should print.
    pub fn show_logs(self) -> bool {
        self != Self::Quiet
    }
    /// True when the daemon should include timing lines in its log.
    pub fn daemon_verbose(self) -> bool {
        self == Self::Verbose
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    Bm25,
    Vector,
    #[default]
    Hybrid,
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bm25" => Ok(Self::Bm25),
            "vector" | "vec" => Ok(Self::Vector),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err(format!("unknown mode '{s}'. Use: bm25, vector, hybrid")),
        }
    }
}
