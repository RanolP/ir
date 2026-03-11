-- FTS sync triggers — only installed when no preprocessor is active.
-- When a preprocessor is active, ir manages FTS inserts/deletes explicitly
-- so that preprocessed text (not raw content) is indexed.

-- Sync FTS on insert
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, path, title, body)
    SELECT new.id, new.path, new.title, c.doc
    FROM content c WHERE c.hash = new.hash;
END;

-- Sync FTS on delete
CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    DELETE FROM documents_fts WHERE rowid = old.id;
END;

-- Sync FTS on update: only re-insert when the document is active.
-- Deactivation (active→0) deletes from FTS without re-inserting.
DROP TRIGGER IF EXISTS documents_au;
CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
    DELETE FROM documents_fts WHERE rowid = old.id;
    INSERT INTO documents_fts(rowid, path, title, body)
    SELECT new.id, new.path, new.title, c.doc
    FROM content c WHERE c.hash = new.hash AND new.active = 1;
END;
