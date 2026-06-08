"""SQLite database management for the MCP server subsystem."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB_PATH = "data/operator_agent.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _resolve_path(db_path: str) -> Path:
    p = Path(db_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def _load_schema() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


class Database:
    """Synchronous SQLite database wrapper."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = _resolve_path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        # Default isolation_level is "" (autocommit per statement) — keep that,
        # we just want explicit commits on writes (existing callers do that).
        # `check_same_thread=False` is required because the MCP server's stdio
        # event-loop and the FastAPI main app both pass this connection to
        # threads; SQLite's thread checks otherwise raise ProgrammingError.
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,  # seconds to wait for the GIL/connection lock
        )
        # ── Concurrency hardening ──────────────────────────────────────
        # WAL mode lets one writer + many readers proceed in parallel, but
        # a second writer still gets SQLITE_BUSY immediately unless we
        # set busy_timeout.  With WAL + busy_timeout the second writer
        # waits for the first to commit instead of failing the request.
        # The MCP server is spawned as a fresh subprocess per tool call
        # (see agent.mcp_client.MCPClient._call_tool), so two writers can
        # land at the same time when the agent main process and the MCP
        # subprocess both touch the SQLite file.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # WAL: safe with NORMAL
        self._conn.execute("PRAGMA busy_timeout=30000")   # wait up to 30s for lock
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_load_schema())
        # 迁移：v2 — 新增 is_optional 列
        try:
            self._conn.execute(
                "ALTER TABLE parameters ADD COLUMN is_optional INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v3 — 新增 src_content 列
        try:
            self._conn.execute(
                "ALTER TABLE parameters ADD COLUMN src_content TEXT"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v4 — 新增 param_relations 表
        try:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS param_relations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id          INTEGER NOT NULL REFERENCES document_versions(id),
                    function_name   TEXT NOT NULL DEFAULT '',
                    relation_type   TEXT NOT NULL,
                    precondition    TEXT NOT NULL DEFAULT '无',
                    description     TEXT NOT NULL,
                    params          TEXT NOT NULL,
                    param_optional  TEXT NOT NULL DEFAULT '{}',
                    source_citation TEXT NOT NULL,
                    created_at      TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_param_relations_doc_id
                    ON param_relations(doc_id);
                """
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v5 — 新增 function_signatures 表
        try:
            self._conn.executescript(
                """
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
                """
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v6 — 新增 platform_support 表
        try:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS platform_support (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id          INTEGER NOT NULL REFERENCES document_versions(id),
                    platform_name   TEXT NOT NULL,
                    is_supported    INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT DEFAULT (datetime('now')),
                    UNIQUE(doc_id, platform_name)
                );
                CREATE INDEX IF NOT EXISTS idx_platform_support_doc_id
                    ON platform_support(doc_id);
                """
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v7 — 新增 is_support_discontinuous 列
        try:
            self._conn.execute(
                "ALTER TABLE parameters ADD COLUMN is_support_discontinuous "
                "TEXT NOT NULL DEFAULT '{\"value\":\"N/A\",\"src_text\":\"\"}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v8 — 新增 array_length 列
        try:
            self._conn.execute(
                "ALTER TABLE parameters ADD COLUMN array_length "
                "TEXT NOT NULL DEFAULT 'N/A'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v9 — 新增 param_desc 列
        try:
            self._conn.execute(
                "ALTER TABLE parameters ADD COLUMN param_desc "
                "TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v10 — 新增 allowed_range_value 列
        try:
            self._conn.execute(
                "ALTER TABLE parameters ADD COLUMN allowed_range_value "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v11 — 新增 return_codes 表
        try:
            self._conn.executescript(
                """
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
                """
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v13 — 新增 dtype_combinations 表
        try:
            self._conn.executescript(
                """
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
                """
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v14 — platform_support 表新增确定性计算 JSON 字段
        try:
            self._conn.execute(
                "ALTER TABLE platform_support ADD COLUMN deterministic_computing "
                "TEXT NOT NULL DEFAULT '{\"value\":\"\",\"src_text\":\"\"}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v14b — 将 determinism 表已有数据迁移到 platform_support，然后删除 determinism 表
        try:
            det_rows = self._conn.execute(
                "SELECT doc_id, product, value, src_text FROM determinism"
            ).fetchall()
            for doc_id, product, value, src_text in det_rows:
                det_json = json.dumps(
                    {"value": "确定性" if value else "非确定性", "src_text": src_text},
                    ensure_ascii=False,
                )
                self._conn.execute(
                    "INSERT INTO platform_support "
                    "(doc_id, platform_name, is_supported, deterministic_computing) "
                    "VALUES (?, ?, 0, ?) "
                    "ON CONFLICT(doc_id, platform_name) DO UPDATE SET "
                    "deterministic_computing = excluded.deterministic_computing",
                    (doc_id, product, det_json),
                )
            self._conn.execute("DROP TABLE IF EXISTS determinism")
        except sqlite3.OperationalError:
            pass
        # 迁移：v15 — 新增 constraints_result 表
        try:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS constraints_result (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id               INTEGER NOT NULL REFERENCES document_versions(id) UNIQUE,
                    operator_name        TEXT NOT NULL,
                    product_support      TEXT NOT NULL DEFAULT '[]',
                    function_explanation TEXT NOT NULL DEFAULT '{}',
                    created_at           TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_constraints_result_doc_id
                    ON constraints_result(doc_id);
                """
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v16 — document_versions 新增 function_explanation_summary 列
        try:
            self._conn.execute(
                "ALTER TABLE document_versions ADD COLUMN function_explanation_summary "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v17 — constraints_result 新增 platform_support 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result ADD COLUMN platform_support "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v18 — constraints_result 新增 function_signature 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result ADD COLUMN function_signature "
                "TEXT NOT NULL DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v19 — parameters 新增 param_constraint 列
        try:
            self._conn.execute(
                "ALTER TABLE parameters ADD COLUMN param_constraint "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v20 — param_relations 新增 relation_object 列
        try:
            self._conn.execute(
                "ALTER TABLE param_relations ADD COLUMN relation_object "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v21 — constraints_result 新增 return_codes 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result ADD COLUMN return_codes "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v22 — constraints_result 新增 deterministic_computing 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result ADD COLUMN deterministic_computing "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v23 — constraints_result 新增 inputs 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result ADD COLUMN inputs "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v24 — constraints_result 新增 outputs 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result ADD COLUMN outputs "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v25 — constraints_result 新增 constraints_in_parameters 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result ADD COLUMN constraints_in_parameters "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v27 — rename constraints_in_param → constraints_in_parameters
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result RENAME COLUMN constraints_in_param "
                "TO constraints_in_parameters"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v26 — constraints_result 新增 dtype_support_description 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result ADD COLUMN dtype_support_description "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v27 — document_versions 新增 json_constraints 列
        try:
            self._conn.execute(
                "ALTER TABLE document_versions ADD COLUMN json_constraints "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v28 — constraints_result 删除 platform_support 列
        try:
            self._conn.execute(
                "ALTER TABLE constraints_result DROP COLUMN platform_support"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v29 — platform_support.deterministic_computing value 从中文改为 true/false
        try:
            self._conn.execute(
                "UPDATE platform_support "
                "SET deterministic_computing = json_set(deterministic_computing, '$.value', 'true') "
                "WHERE json_extract(deterministic_computing, '$.value') = '确定性'"
            )
            self._conn.execute(
                "UPDATE platform_support "
                "SET deterministic_computing = json_set(deterministic_computing, '$.value', 'false') "
                "WHERE json_extract(deterministic_computing, '$.value') = '非确定性'"
            )
        except sqlite3.OperationalError:
            pass
        # 迁移：v29b — constraints_result.deterministic_computing 同步转换
        try:
            # constraints_result.deterministic_computing is a JSON object keyed by platform_name
            # e.g. {"Atlas A2": {"value": "确定性", "src_text": "..."}}
            # Use json_patch-style update: iterate keys is hard in SQL, so do it in Python
            rows = self._conn.execute(
                "SELECT id, deterministic_computing FROM constraints_result "
                "WHERE deterministic_computing != '{}'"
            ).fetchall()
            for row_id, dc_raw in rows:
                try:
                    dc = json.loads(dc_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                changed = False
                for plat, det in dc.items():
                    if isinstance(det, dict):
                        v = det.get("value", "")
                        if v == "确定性":
                            det["value"] = "true"
                            changed = True
                        elif v == "非确定性":
                            det["value"] = "false"
                            changed = True
                if changed:
                    self._conn.execute(
                        "UPDATE constraints_result SET deterministic_computing = ? WHERE id = ?",
                        (json.dumps(dc, ensure_ascii=False), row_id),
                    )
        except sqlite3.OperationalError:
            pass
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(parameters)").fetchall()}
        if "src_content" not in existing:
            self._conn.execute("ALTER TABLE parameters ADD COLUMN src_content TEXT")
            self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


_db: Database | None = None


def get_db(db_path: str | None = None) -> Database:
    global _db
    if _db is None:
        resolved = db_path or os.environ.get("DATABASE_PATH", DEFAULT_DB_PATH)
        _db = Database(resolved)
        _db.connect()
    return _db
