// Document retrieval by path -- shared by CLI and MCP.
// Supports exact, suffix, and substring matching with vault-root path resolution.

use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};

use crate::config::{Config, collection_db_path};
use crate::db;
use crate::error::Result;
use crate::types::Collection;

// ── output types ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct DocContent {
    pub collection: String,
    pub path: String,
    pub title: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MultiGetResult {
    pub found: Vec<DocContent>,
    /// Paths that had no match in any collection
    pub not_found: Vec<String>,
}

// ── SQL ──────────────────────────────────────────────────────────────────────

const SQL_EXACT: &str = "SELECT d.path, d.title, c.doc \
    FROM documents d JOIN content c ON d.hash = c.hash \
    WHERE d.path = ?1 AND d.active = 1 LIMIT 1";
// ^ ESCAPE clause required so literal % and _ in paths don't act as LIKE wildcards.
const SQL_LIKE_ESCAPED: &str = "SELECT d.path, d.title, c.doc \
    FROM documents d JOIN content c ON d.hash = c.hash \
    WHERE d.path LIKE ?1 ESCAPE '\\' AND d.active = 1 LIMIT 1";

/// Escape `%` and `_` so LIKE treats them as literals.
fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

// ── public API ───────────────────────────────────────────────────────────────

pub fn open_readonly(path: &std::path::Path) -> std::result::Result<Connection, rusqlite::Error> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let _ = conn.execute_batch("PRAGMA busy_timeout = 5000;");
    Ok(conn)
}

pub fn fetch_document(path: &str, collection_filter: &[String]) -> Result<Option<DocContent>> {
    let config = Config::load()?;
    fetch_document_with_config(path, collection_filter, &config)
}

pub fn fetch_document_with_config(
    path: &str,
    collection_filter: &[String],
    config: &Config,
) -> Result<Option<DocContent>> {
    let cols: Vec<&Collection> = if collection_filter.is_empty() {
        config.collections.iter().collect()
    } else {
        config.collections.iter().filter(|c| collection_filter.contains(&c.name)).collect()
    };

    db::ensure_sqlite_vec();

    // Try vault-root prefix first: "CollectionDir/rel/path" → search CollectionDir's DB with rel/path.
    if let Some((col, stripped)) = resolve_vault_root_path(path, &cols) {
        let db_path = collection_db_path(&col.name);
        match open_readonly(&db_path) {
            Ok(conn) => {
                if let Some(doc) = lookup_in_conn(&conn, &col.name, &stripped)? {
                    return Ok(Some(doc));
                }
            }
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::CannotOpen => {}
            Err(e) => return Err(e.into()),
        }
    }

    // Fallback: try all collections with the original path (including any vault-root collection,
    // in case the path is stored verbatim with the prefix inside that collection).
    for col in &cols {
        let db_path = collection_db_path(&col.name);
        let conn = match open_readonly(&db_path) {
            Ok(c) => c,
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::CannotOpen => continue,
            Err(e) => return Err(e.into()),
        };
        if let Some(doc) = lookup_in_conn(&conn, &col.name, path)? {
            return Ok(Some(doc));
        }
    }
    Ok(None)
}

/// Try exact, suffix, then substring path match. Stops at first hit.
pub fn lookup_in_conn(conn: &Connection, collection: &str, path: &str) -> Result<Option<DocContent>> {
    if path.is_empty() {
        return Ok(None);
    }
    // ^ Escape LIKE wildcards so literal % and _ in paths don't cause false positives.
    let escaped = escape_like(path);
    let suffix = format!("%/{escaped}");
    let substr = format!("%{escaped}%");
    let queries: &[(&str, &str)] = &[
        (SQL_EXACT, path),
        (SQL_LIKE_ESCAPED, &suffix),
        (SQL_LIKE_ESCAPED, &substr),
    ];
    for (sql, param) in queries {
        let row = conn.query_row(sql, rusqlite::params![param], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        });
        match row {
            Ok((doc_path, title, content)) => {
                return Ok(Some(DocContent {
                    collection: collection.to_string(),
                    path: doc_path,
                    title,
                    content,
                }));
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(None)
}

// ── vault-root path resolution ───────────────────────────────────────────────

/// If path starts with a collection's directory name (the last component of its
/// absolute path), return that collection and the remainder after stripping.
///
/// Example: path "0. PeriodicNotes/2026/file.md", collection with
/// path "/vault/0. PeriodicNotes" -> Some((col, "2026/file.md"))
fn resolve_vault_root_path<'a>(
    path: &str,
    collections: &[&'a Collection],
) -> Option<(&'a Collection, String)> {
    let (first, rest) = path.split_once('/')?;
    if rest.is_empty() {
        return None;
    }
    for col in collections {
        // ^ skip collections whose path has no usable dir component (e.g. root "/" or non-UTF-8)
        let col_path = std::path::Path::new(&col.path);
        let dir_name = match col_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if first == dir_name {
            return Some((col, rest.to_string()));
        }
    }
    None
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_col(name: &str, path: &str) -> Collection {
        Collection {
            name: name.to_string(),
            path: path.to_string(),
            globs: vec![],
            excludes: vec![],
            description: None,
            preprocessor: None,
        }
    }

    fn open_test_db() -> Connection {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(include_str!("db/schema_base.sql")).unwrap();
        conn
    }

    fn insert_doc(conn: &Connection, path: &str, title: &str, content: &str) {
        let hash = format!("hash_{}", path.replace('/', "_"));
        conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, '2026-01-01')",
            rusqlite::params![hash, content],
        ).unwrap();
        conn.execute(
            "INSERT INTO documents (path, title, hash, created_at, modified_at, active) \
             VALUES (?1, ?2, ?3, '2026-01-01', '2026-01-01', 1)",
            rusqlite::params![path, title, hash],
        ).unwrap();
    }

    fn insert_inactive_doc(conn: &Connection, path: &str, title: &str, content: &str) {
        let hash = format!("hash_{}", path.replace('/', "_"));
        conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, '2026-01-01')",
            rusqlite::params![hash, content],
        ).unwrap();
        conn.execute(
            "INSERT INTO documents (path, title, hash, created_at, modified_at, active) \
             VALUES (?1, ?2, ?3, '2026-01-01', '2026-01-01', 0)",
            rusqlite::params![path, title, hash],
        ).unwrap();
    }

    // ── resolve_vault_root_path ──────────────────────────────────────────────

    #[test]
    fn vault_root_strips_prefix() {
        let periodic = test_col("periodic", "/vault/0. PeriodicNotes");
        let cols: Vec<&Collection> = vec![&periodic];
        let (col, rest) = resolve_vault_root_path(
            "0. PeriodicNotes/2026/Daily/04/file.md", &cols
        ).unwrap();
        assert_eq!(col.name, "periodic");
        assert_eq!(rest, "2026/Daily/04/file.md");
    }

    #[test]
    fn vault_root_no_match_returns_none() {
        let periodic = test_col("periodic", "/vault/0. PeriodicNotes");
        let cols: Vec<&Collection> = vec![&periodic];
        assert!(resolve_vault_root_path("other/2026/file.md", &cols).is_none());
    }

    #[test]
    fn vault_root_no_slash_returns_none() {
        let periodic = test_col("periodic", "/vault/0. PeriodicNotes");
        let cols: Vec<&Collection> = vec![&periodic];
        assert!(resolve_vault_root_path("just-a-filename.md", &cols).is_none());
    }

    #[test]
    fn vault_root_multiple_collections_picks_correct() {
        let periodic = test_col("periodic", "/vault/0. PeriodicNotes");
        let projects = test_col("projects", "/vault/1. Projects");
        let cols: Vec<&Collection> = vec![&periodic, &projects];
        let (col, rest) = resolve_vault_root_path(
            "1. Projects/myproject/README.md", &cols
        ).unwrap();
        assert_eq!(col.name, "projects");
        assert_eq!(rest, "myproject/README.md");
    }

    #[test]
    fn vault_root_case_sensitive() {
        let notes = test_col("notes", "/vault/Notes");
        let cols: Vec<&Collection> = vec![&notes];
        // lowercase "notes" should NOT match "Notes"
        assert!(resolve_vault_root_path("notes/file.md", &cols).is_none());
    }

    #[test]
    fn vault_root_nested_path_only_strips_first_component() {
        let col = test_col("deep", "/vault/a/b/c");
        let cols: Vec<&Collection> = vec![&col];
        // file_name() of "/vault/a/b/c" is "c"
        let (matched, rest) = resolve_vault_root_path("c/file.md", &cols).unwrap();
        assert_eq!(matched.name, "deep");
        assert_eq!(rest, "file.md");
    }

    #[test]
    fn vault_root_with_spaces_in_dirname() {
        let col = test_col("periodic", "/vault/0. Periodic Notes");
        let cols: Vec<&Collection> = vec![&col];
        let (matched, rest) = resolve_vault_root_path(
            "0. Periodic Notes/2026/file.md", &cols
        ).unwrap();
        assert_eq!(matched.name, "periodic");
        assert_eq!(rest, "2026/file.md");
    }

    // ── lookup_in_conn ───────────────────────────────────────────────────────

    #[test]
    fn lookup_exact_match() {
        let conn = open_test_db();
        insert_doc(&conn, "2026/Daily/04/file.md", "File", "hello world");
        let doc = lookup_in_conn(&conn, "test", "2026/Daily/04/file.md").unwrap().unwrap();
        assert_eq!(doc.path, "2026/Daily/04/file.md");
        assert_eq!(doc.title, "File");
        assert_eq!(doc.content, "hello world");
        assert_eq!(doc.collection, "test");
    }

    #[test]
    fn lookup_suffix_match() {
        let conn = open_test_db();
        insert_doc(&conn, "2026/Daily/04/file.md", "File", "content");
        // Suffix: requesting just "04/file.md" should match via %/04/file.md
        let doc = lookup_in_conn(&conn, "test", "04/file.md").unwrap().unwrap();
        assert_eq!(doc.path, "2026/Daily/04/file.md");
    }

    #[test]
    fn lookup_substring_match() {
        let conn = open_test_db();
        insert_doc(&conn, "2026/Daily/04/file.md", "File", "content");
        // Substring: partial match via %Daily%
        let doc = lookup_in_conn(&conn, "test", "Daily/04/file").unwrap().unwrap();
        assert_eq!(doc.path, "2026/Daily/04/file.md");
    }

    #[test]
    fn lookup_no_match() {
        let conn = open_test_db();
        insert_doc(&conn, "2026/Daily/04/file.md", "File", "content");
        assert!(lookup_in_conn(&conn, "test", "nonexistent.md").unwrap().is_none());
    }

    #[test]
    fn lookup_prefers_exact_over_suffix() {
        let conn = open_test_db();
        insert_doc(&conn, "file.md", "Exact", "exact content");
        insert_doc(&conn, "subdir/file.md", "Suffix", "suffix content");
        // "file.md" should exact-match, not suffix-match "subdir/file.md"
        let doc = lookup_in_conn(&conn, "test", "file.md").unwrap().unwrap();
        assert_eq!(doc.title, "Exact");
    }

    #[test]
    fn lookup_skips_inactive() {
        let conn = open_test_db();
        insert_inactive_doc(&conn, "file.md", "Inactive", "old content");
        assert!(lookup_in_conn(&conn, "test", "file.md").unwrap().is_none());
    }

    #[test]
    fn lookup_with_sql_wildcard_in_path() {
        let conn = open_test_db();
        insert_doc(&conn, "notes/100% done.md", "Percent", "content");
        // Path with literal % should still work for exact match
        let doc = lookup_in_conn(&conn, "test", "notes/100% done.md").unwrap().unwrap();
        assert_eq!(doc.title, "Percent");
    }

    #[test]
    fn lookup_with_underscore_in_path() {
        let conn = open_test_db();
        insert_doc(&conn, "my_notes/file.md", "Underscore", "content");
        // _ is a LIKE wildcard but exact match should take priority
        let doc = lookup_in_conn(&conn, "test", "my_notes/file.md").unwrap().unwrap();
        assert_eq!(doc.title, "Underscore");
    }

    // ── LIKE injection edge cases ────────────────────────────────────────────

    #[test]
    fn like_percent_in_suffix_tier_no_false_positive() {
        let conn = open_test_db();
        insert_doc(&conn, "notes/100% done.md", "Percent", "right");
        insert_doc(&conn, "notes/100X done.md", "Wrong", "wrong");
        // Search "100% done.md" (no exact match). Suffix LIKE must not treat
        // the literal % as a wildcard matching "100X done.md".
        let doc = lookup_in_conn(&conn, "test", "100% done.md").unwrap().unwrap();
        assert_eq!(doc.title, "Percent");
    }

    #[test]
    fn like_underscore_in_suffix_tier_no_false_positive() {
        let conn = open_test_db();
        insert_doc(&conn, "notes/a_b.md", "Underscore", "right");
        insert_doc(&conn, "notes/axb.md", "Wrong", "wrong");
        // _ in LIKE matches any single char. Must not match "axb.md".
        let doc = lookup_in_conn(&conn, "test", "a_b.md").unwrap().unwrap();
        assert_eq!(doc.title, "Underscore");
    }

    // ── empty / degenerate paths ─────────────────────────────────────────────

    #[test]
    fn lookup_empty_path_returns_none() {
        let conn = open_test_db();
        insert_doc(&conn, "file.md", "File", "content");
        assert!(lookup_in_conn(&conn, "test", "").unwrap().is_none());
    }

    #[test]
    fn vault_root_dirname_with_trailing_slash_returns_none() {
        // Path "Notes/" splits to ("Notes", ""), rest is empty -> None
        let col = test_col("notes", "/vault/Notes");
        let cols: Vec<&Collection> = vec![&col];
        assert!(resolve_vault_root_path("Notes/", &cols).is_none());
    }

    // ── vault-root + collection filter interaction ───────────────────────────

    #[test]
    fn vault_root_duplicate_basename_picks_first() {
        let a = test_col("notes-a", "/vault-a/Notes");
        let b = test_col("notes-b", "/vault-b/Notes");
        let cols: Vec<&Collection> = vec![&a, &b];
        let (col, _) = resolve_vault_root_path("Notes/file.md", &cols).unwrap();
        assert_eq!(col.name, "notes-a");
    }

    #[test]
    fn vault_root_trailing_slash_in_collection_path() {
        // Path::file_name() strips trailing slash on unix
        let col = test_col("notes", "/vault/Notes/");
        let cols: Vec<&Collection> = vec![&col];
        let (matched, rest) = resolve_vault_root_path("Notes/file.md", &cols).unwrap();
        assert_eq!(matched.name, "notes");
        assert_eq!(rest, "file.md");
    }

    // ── suffix vs substring tier semantics ───────────────────────────────────

    #[test]
    fn suffix_requires_preceding_slash() {
        let conn = open_test_db();
        insert_doc(&conn, "myfile.md", "My", "content");
        // "file.md" has no exact match. Suffix "%/file.md" won't match "myfile.md"
        // (no slash before "file.md"). Substring "%file.md%" will match.
        let doc = lookup_in_conn(&conn, "test", "file.md").unwrap().unwrap();
        assert_eq!(doc.path, "myfile.md");
    }

    // ── unicode paths ────────────────────────────────────────────────────────

    #[test]
    fn lookup_cjk_filename_via_suffix() {
        let conn = open_test_db();
        insert_doc(&conn, "日本語/ファイル.md", "CJK", "content");
        let doc = lookup_in_conn(&conn, "test", "ファイル.md").unwrap().unwrap();
        assert_eq!(doc.path, "日本語/ファイル.md");
    }
}
