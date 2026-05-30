"""SQLite database management for the MCP server subsystem."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = "data/operator_agent.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _load_schema() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


class Database:
    """Synchronous SQLite database wrapper."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
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
