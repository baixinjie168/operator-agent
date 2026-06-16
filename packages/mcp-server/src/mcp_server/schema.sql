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
    function_explanation_summary TEXT NOT NULL DEFAULT '{}',
    json_constraints TEXT NOT NULL DEFAULT '{}',
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
    dtype_desc      TEXT,
    dformat_desc    TEXT,
    shape           TEXT,
    is_optional     INTEGER NOT NULL DEFAULT 0,
    is_support_discontinuous TEXT NOT NULL DEFAULT '{"value":"N/A","src_text":""}',
    array_length    TEXT NOT NULL DEFAULT 'N/A',
    param_desc      TEXT NOT NULL DEFAULT '',
    allowed_range_value TEXT NOT NULL DEFAULT '[]',
    param_constraint    TEXT NOT NULL DEFAULT '{}',
    llm_description     TEXT NOT NULL DEFAULT '',
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
    task_type       TEXT,
    task_name       TEXT,
    parent_task_id  TEXT REFERENCES pipeline_runs(run_id),
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

CREATE TABLE IF NOT EXISTS param_relations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          INTEGER NOT NULL REFERENCES document_versions(id),
    function_name   TEXT NOT NULL DEFAULT '',
    relation_type   TEXT NOT NULL,
    platform        TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL,
    params          TEXT NOT NULL,
    param_optional  TEXT NOT NULL DEFAULT '{}',
    source_citation TEXT NOT NULL,
    relation_object TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_param_relations_doc_id
    ON param_relations(doc_id);

CREATE TABLE IF NOT EXISTS function_signatures (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id           INTEGER NOT NULL REFERENCES document_versions(id),
    function_name    TEXT NOT NULL,
    return_type      TEXT NOT NULL DEFAULT '',
    parameters       TEXT NOT NULL DEFAULT '[]',
    full_signature   TEXT NOT NULL DEFAULT '',
    raw_code         TEXT NOT NULL DEFAULT '',
    created_at       TEXT DEFAULT (datetime('now')),
    UNIQUE(doc_id, function_name)
);

CREATE INDEX IF NOT EXISTS idx_function_signatures_doc_id
    ON function_signatures(doc_id);

CREATE TABLE IF NOT EXISTS platform_support (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id                   INTEGER NOT NULL REFERENCES document_versions(id),
    platform_name            TEXT NOT NULL,
    is_supported             INTEGER NOT NULL DEFAULT 0,
    deterministic_computing  TEXT NOT NULL DEFAULT '{"value":"","src_text":""}',
    created_at               TEXT DEFAULT (datetime('now')),
    UNIQUE(doc_id, platform_name)
);

CREATE INDEX IF NOT EXISTS idx_platform_support_doc_id
    ON platform_support(doc_id);

CREATE TABLE IF NOT EXISTS return_codes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          INTEGER NOT NULL REFERENCES document_versions(id),
    function_name   TEXT NOT NULL DEFAULT '',
    return_value    TEXT NOT NULL,
    error_code      INTEGER NOT NULL,
    descriptions    TEXT NOT NULL DEFAULT '[]',
    source_citation TEXT NOT NULL DEFAULT '',
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(doc_id, function_name, return_value, error_code)
);

CREATE INDEX IF NOT EXISTS idx_return_codes_doc_id
    ON return_codes(doc_id);

CREATE TABLE IF NOT EXISTS dtype_combinations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          INTEGER NOT NULL REFERENCES document_versions(id),
    function_name   TEXT NOT NULL DEFAULT '',
    platform        TEXT NOT NULL DEFAULT '通用',
    combo           TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dtype_combos_doc_id
    ON dtype_combinations(doc_id);
CREATE INDEX IF NOT EXISTS idx_dtype_combos_doc_fn
    ON dtype_combinations(doc_id, function_name);

CREATE TABLE IF NOT EXISTS constraints_result (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id                     INTEGER NOT NULL REFERENCES document_versions(id) UNIQUE,
    operator_name              TEXT NOT NULL,
    product_support            TEXT NOT NULL DEFAULT '[]',
    function_explanation       TEXT NOT NULL DEFAULT '{}',
    function_signature         TEXT NOT NULL DEFAULT '',
    return_codes               TEXT NOT NULL DEFAULT '[]',
    deterministic_computing    TEXT NOT NULL DEFAULT '{}',
    inputs                     TEXT NOT NULL DEFAULT '{}',
    outputs                    TEXT NOT NULL DEFAULT '{}',
    constraints_in_parameters  TEXT NOT NULL DEFAULT '{}',
    dtype_support_description  TEXT NOT NULL DEFAULT '{}',
    created_at                 TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_constraints_result_doc_id
    ON constraints_result(doc_id);

CREATE TABLE IF NOT EXISTS test_cases (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id           TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    operator_name     TEXT NOT NULL,
    case_index        INTEGER NOT NULL,
    case_name         TEXT NOT NULL,
    case_data         TEXT NOT NULL,
    constraint_doc_id INTEGER REFERENCES document_versions(id),
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_test_cases_task ON test_cases(task_id);
CREATE INDEX IF NOT EXISTS idx_test_cases_operator ON test_cases(operator_name);
CREATE INDEX IF NOT EXISTS idx_test_cases_constraint_doc ON test_cases(constraint_doc_id);

CREATE TABLE IF NOT EXISTS exec_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    case_id             INTEGER NOT NULL REFERENCES test_cases(id),
    operator_name       TEXT NOT NULL,
    passed              INTEGER NOT NULL,
    cpu_precision_passed INTEGER,
    precision_detail    TEXT,
    actual_json         TEXT,
    error_message       TEXT,
    cpu_reference_code  TEXT,
    duration_ms         INTEGER,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_exec_results_task ON exec_results(task_id);
CREATE INDEX IF NOT EXISTS idx_exec_results_case ON exec_results(case_id);
CREATE INDEX IF NOT EXISTS idx_exec_results_operator ON exec_results(operator_name);

CREATE TABLE IF NOT EXISTS tasks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    total_count     INTEGER NOT NULL,
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count    INTEGER NOT NULL DEFAULT 0,
    upload_dir      TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS task_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         INTEGER NOT NULL REFERENCES tasks(id),
    seq             INTEGER NOT NULL,
    operator_name   TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    doc_id          INTEGER,
    error           TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_task_items_task_id
    ON task_items(task_id);
