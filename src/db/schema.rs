// Per-collection SQLite schema.
// Each collection gets its own .sqlite file — no collection column needed.

use crate::error::Result;
use rusqlite::Connection;

const SCHEMA_VERSION: i64 = 1;

pub fn init(conn: &Connection, collection_name: &str, has_preprocessor: bool) -> Result<()> {
    conn.execute_batch(include_str!("schema_base.sql"))?;

    if has_preprocessor {
        // Drop any pre-existing triggers; FTS is managed explicitly by the index pipeline.
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS documents_ai;
             DROP TRIGGER IF EXISTS documents_ad;
             DROP TRIGGER IF EXISTS documents_au;",
        )?;
    } else {
        conn.execute_batch(include_str!("schema_triggers.sql"))?;
    }

    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION.to_string()],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('collection', ?1)",
        [collection_name],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('has_preprocessor', ?1)",
        [if has_preprocessor { "1" } else { "0" }],
    )?;
    Ok(())
}
