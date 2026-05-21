CREATE TABLE IF NOT EXISTS operators (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    source_url  TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS document_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_id     INTEGER NOT NULL REFERENCES operators(id),
    version         INTEGER NOT NULL DEFAULT 1,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    parsed_data     TEXT,
    product_support TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(operator_id, version)
);

CREATE TABLE IF NOT EXISTS parameters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          INTEGER NOT NULL REFERENCES document_versions(id),
    function_name   TEXT NOT NULL,
    param_name      TEXT NOT NULL,
    param_type      TEXT NOT NULL DEFAULT '',
    direction       TEXT NOT NULL DEFAULT 'input',
    description     TEXT,
    usage_notes     TEXT,
    dtype_desc      TEXT,
    dformat_desc    TEXT,
    shape           TEXT,
    memory_desc     TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(doc_id, function_name, param_name)
);

CREATE INDEX IF NOT EXISTS idx_parameters_doc_id
    ON parameters(doc_id);
