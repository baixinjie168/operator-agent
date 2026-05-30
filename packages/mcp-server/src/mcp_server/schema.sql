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
    src_content      TEXT,
    description     TEXT,
    usage_notes     TEXT,
    dtype_desc      TEXT,
    dformat_desc    TEXT,
    shape           TEXT,
    memory_desc     TEXT,
    is_optional     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(doc_id, function_name, param_name)
);

CREATE INDEX IF NOT EXISTS idx_parameters_doc_id
    ON parameters(doc_id);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL UNIQUE,
    operator_id     INTEGER REFERENCES operators(id),
    doc_id          INTEGER REFERENCES document_versions(id),
    operator_name   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    content_hash    TEXT NOT NULL,
    result_json     TEXT,
    error           TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    seq         INTEGER NOT NULL,
    event_type  TEXT NOT NULL,
    data_json   TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_operator
    ON pipeline_runs(operator_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_run_id
    ON pipeline_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_events_run
    ON pipeline_events(run_id, seq);
