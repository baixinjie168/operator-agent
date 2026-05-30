"""Direct DB access for runtime persistence — bypasses MCP subprocess.

MCP is for document parsing operations.  Pipeline run/event persistence is simple SQLite.
"""

from __future__ import annotations

import json

from mcp_server.db import get_db


def create_run(run_id: str, operator_name: str, content_hash: str) -> None:
    db = get_db()
    db.conn.execute(
        "INSERT INTO pipeline_runs (run_id, operator_name, status, content_hash) VALUES (?, ?, 'running', ?)",
        (run_id, operator_name, content_hash),
    )
    db.conn.commit()


def update_run_doc_id(run_id: str, doc_id: int) -> None:
    """Update the doc_id on a pipeline run — called early, right after init_doc completes."""
    db = get_db()
    db.conn.execute(
        "UPDATE pipeline_runs SET doc_id = ? WHERE run_id = ?",
        (doc_id, run_id),
    )
    db.conn.commit()


def complete_run(run_id: str, result: dict, error: str | None = None, doc_id: int | None = None) -> None:
    db = get_db()
    if error:
        db.conn.execute(
            "UPDATE pipeline_runs SET status = 'failed', error = ?, completed_at = datetime('now') WHERE run_id = ?",
            (error, run_id),
        )
    else:
        db.conn.execute(
            "UPDATE pipeline_runs SET status = 'completed', result_json = ?, doc_id = ?, completed_at = datetime('now') WHERE run_id = ?",
            (json.dumps(result, ensure_ascii=False), doc_id, run_id),
        )
    db.conn.commit()


def save_events(run_id: str, events: list[dict]) -> None:
    db = get_db()
    data = [(run_id, e["seq"], e["event_type"], json.dumps(e["data"], ensure_ascii=False)) for e in events]
    db.conn.executemany(
        "INSERT INTO pipeline_events (run_id, seq, event_type, data_json) VALUES (?, ?, ?, ?)",
        data,
    )
    db.conn.commit()


def query_params_by_doc_id(doc_id: int) -> list[dict]:
    """Query all parameters for a specific document version by doc_id.

    Returns the key fields needed for the frontend parameter detail table.
    """
    db = get_db()
    rows = db.conn.execute(
        "SELECT id, param_name, param_type, direction, description, "
        "dtype_desc, dformat_desc, shape, memory_desc, src_content "
        "FROM parameters WHERE doc_id = ? "
        "ORDER BY id",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "param_name": r[1],
            "param_type": r[2],
            "direction": r[3],
            "description": r[4],
            "dtype_desc": r[5],
            "dformat_desc": r[6],
            "shape": r[7],
            "memory_desc": r[8],
            "src_content": r[9],
        }
        for r in rows
    ]


def update_param_src_content(param_id: int, src_content: str) -> bool:
    """Update the src_content field of a single parameter by its primary key."""
    db = get_db()
    cursor = db.conn.execute(
        "UPDATE parameters SET src_content = ? WHERE id = ?",
        (src_content, param_id),
    )
    db.conn.commit()
    return cursor.rowcount > 0


def query_runs(operator_id: int | None = None, limit: int = 20) -> list[dict]:
    db = get_db()
    if operator_id:
        rows = db.conn.execute(
            "SELECT * FROM pipeline_runs WHERE operator_id = ? ORDER BY created_at DESC LIMIT ?",
            (operator_id, limit),
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    cols = ["id", "run_id", "operator_id", "doc_id", "operator_name", "status", "content_hash",
            "result_json", "error", "created_at", "completed_at"]
    return [dict(zip(cols, r)) for r in rows]


def query_run(run_id: str) -> dict | None:
    db = get_db()
    row = db.conn.execute("SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    cols = ["id", "run_id", "operator_id", "doc_id", "operator_name", "status", "content_hash",
            "result_json", "error", "created_at", "completed_at"]
    return dict(zip(cols, row))


def query_events(run_id: str, since_seq: int = 0) -> list[dict]:
    db = get_db()
    rows = db.conn.execute(
        "SELECT seq, event_type, data_json, created_at FROM pipeline_events WHERE run_id = ? AND seq > ? ORDER BY seq",
        (run_id, since_seq),
    ).fetchall()
    return [{"seq": r[0], "event_type": r[1], "data": json.loads(r[2]), "ts": r[3]} for r in rows]
