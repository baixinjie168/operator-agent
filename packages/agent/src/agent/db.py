"""Direct DB access for runtime persistence — bypasses MCP subprocess.

MCP is for document parsing operations.  Pipeline run/event persistence is simple SQLite.
"""

from __future__ import annotations

import json
import logging

from mcp_server.db import get_db

logger = logging.getLogger(__name__)

_TASK_TYPE_LABELS = {
    "constraint_extract": "约束提取",
    "case_generate": "用例生成",
    "test_execute": "测试执行",
}


def generate_task_name(operator_name: str, task_type: str) -> str:
    """Generate a task name like 'aclnnAdaLayerNorm 约束提取 #3'."""
    label = _TASK_TYPE_LABELS.get(task_type, task_type)
    db = get_db()
    row = db.conn.execute(
        "SELECT COUNT(*) FROM pipeline_runs WHERE operator_name = ? AND task_type = ?",
        (operator_name, task_type),
    ).fetchone()
    seq = (row[0] if row else 0) + 1
    return f"{operator_name} {label} #{seq}"


def create_run(
    run_id: str,
    operator_name: str,
    content_hash: str,
    task_type: str | None = None,
    task_name: str | None = None,
    parent_task_id: str | None = None,
) -> None:
    db = get_db()
    if task_name is None and task_type is not None:
        task_name = generate_task_name(operator_name, task_type)
    db.conn.execute(
        "INSERT INTO pipeline_runs (run_id, operator_name, status, content_hash, task_type, task_name, parent_task_id) "
        "VALUES (?, ?, 'running', ?, ?, ?, ?)",
        (run_id, operator_name, content_hash, task_type, task_name, parent_task_id),
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
        "SELECT id, function_name, param_name, param_type, direction, llm_description, "
        "dtype_desc, dformat_desc, shape, src_content, is_optional, "
        "is_support_discontinuous, array_length, allowed_range_value, param_constraint "
        "FROM parameters WHERE doc_id = ? "
        "ORDER BY id",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "function_name": r[1],
            "param_name": r[2],
            "param_type": r[3],
            "direction": r[4],
            "description": r[5],
            "dtype_desc": r[6],
            "dformat_desc": r[7],
            "shape": r[8],
            "src_content": r[9],
            "is_optional": r[10],
            "is_support_discontinuous": r[11],
            "array_length": r[12],
            "allowed_range_value": r[13],
            "param_constraint": r[14],
        }
        for r in rows
    ]


def query_relations_by_doc_id(doc_id: int) -> list[dict]:
    """Query all param_relations for a specific document version by doc_id."""
    db = get_db()
    rows = db.conn.execute(
        "SELECT id, function_name, relation_type, platform, description, "
        "params, source_citation, relation_object "
        "FROM param_relations WHERE doc_id = ? "
        "ORDER BY id",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "function_name": r[1],
            "relation_type": r[2],
            "platform": r[3],
            "description": r[4],
            "params": json.loads(r[5]) if r[5] else [],
            "source_citation": r[6],
            "relation_object": json.loads(r[7]) if r[7] else {},
        }
        for r in rows
    ]


def query_json_constraints_by_doc_id(doc_id: int) -> dict | None:
    """Query json_constraints from document_versions by doc_id."""
    db = get_db()
    row = db.conn.execute(
        "SELECT json_constraints, operator_id FROM document_versions WHERE id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        return None
    jc_raw = row[0] or "{}"
    try:
        jc = json.loads(jc_raw) if isinstance(jc_raw, str) else jc_raw
    except (json.JSONDecodeError, TypeError):
        jc = {}
    return {"json_constraints": jc, "doc_id": doc_id}


def update_param_src_content(param_id: int, src_content: str) -> bool:
    """Update the src_content field of a single parameter by its primary key."""
    db = get_db()
    cursor = db.conn.execute(
        "UPDATE parameters SET src_content = ? WHERE id = ?",
        (src_content, param_id),
    )
    db.conn.commit()
    return cursor.rowcount > 0


_RUN_COLS = [
    "id", "run_id", "operator_id", "doc_id", "operator_name", "status",
    "content_hash", "result_json", "error", "task_type", "task_name",
    "parent_task_id", "created_at", "completed_at",
]

_RUN_SELECT = ", ".join(_RUN_COLS)


def query_runs(
    operator_id: int | None = None,
    operator_name: str | None = None,
    task_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    db = get_db()
    conditions = []
    params: list = []
    if operator_id:
        conditions.append("operator_id = ?")
        params.append(operator_id)
    if operator_name:
        conditions.append("operator_name = ?")
        params.append(operator_name)
    if task_type:
        conditions.append("task_type = ?")
        params.append(task_type)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    rows = db.conn.execute(
        f"SELECT {_RUN_SELECT} FROM pipeline_runs{where} ORDER BY created_at DESC LIMIT ?",
        params,
    ).fetchall()
    return [dict(zip(_RUN_COLS, r, strict=False)) for r in rows]


def query_run(run_id: str) -> dict | None:
    db = get_db()
    row = db.conn.execute(
        f"SELECT {_RUN_SELECT} FROM pipeline_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if not row:
        return None
    return dict(zip(_RUN_COLS, row, strict=False))


def query_events(run_id: str, since_seq: int = 0) -> list[dict]:
    db = get_db()
    rows = db.conn.execute(
        "SELECT seq, event_type, data_json, created_at FROM pipeline_events WHERE run_id = ? AND seq > ? ORDER BY seq",
        (run_id, since_seq),
    ).fetchall()
    return [{"seq": r[0], "event_type": r[1], "data": json.loads(r[2]), "ts": r[3]} for r in rows]


def find_parent_task(operator_name: str, parent_type: str) -> str | None:
    """Find the latest completed task of the given type for the operator."""
    db = get_db()
    row = db.conn.execute(
        "SELECT run_id FROM pipeline_runs "
        "WHERE operator_name = ? AND task_type = ? AND status = 'completed' "
        "ORDER BY created_at DESC LIMIT 1",
        (operator_name, parent_type),
    ).fetchone()
    return row[0] if row else None


def get_task_chain(run_id: str) -> list[dict]:
    """Get the full dependency chain: task → parent → grandparent → ..."""
    chain = []
    current_id = run_id
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        task = query_run(current_id)
        if not task:
            break
        chain.append(task)
        current_id = task.get("parent_task_id")
    return chain


def delete_task(run_id: str) -> dict:
    """Delete a task and all its descendant tasks with cascading data."""
    chain_ids = []
    queue = [run_id]
    visited = set()
    db = get_db()

    while queue:
        tid = queue.pop(0)
        if tid in visited:
            continue
        visited.add(tid)
        chain_ids.append(tid)
        children = db.conn.execute(
            "SELECT run_id FROM pipeline_runs WHERE parent_task_id = ?",
            (tid,),
        ).fetchall()
        for child in children:
            queue.append(child[0])

    deleted = 0
    for tid in reversed(chain_ids):
        db.conn.execute("DELETE FROM exec_results WHERE task_id = ?", (tid,))
        db.conn.execute("DELETE FROM test_cases WHERE task_id = ?", (tid,))
        db.conn.execute("DELETE FROM pipeline_events WHERE run_id = ?", (tid,))
        db.conn.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (tid,))
        deleted += 1

    db.conn.commit()
    logger.info("Deleted task chain: %d tasks starting from %s", deleted, run_id)
    return {"deleted_tasks": deleted, "task_ids": chain_ids}


# ── Test Cases ──────────────────────────────────────────────────────────────

def save_test_cases(
    task_id: str,
    operator_name: str,
    cases: list[dict],
    constraint_doc_id: int | None = None,
) -> dict:
    """Save test cases as individual records, one row per case."""
    db = get_db()
    for idx, case in enumerate(cases):
        case_name = case.get("name", f"{operator_name}_case_{idx}")
        case_data = json.dumps(case, ensure_ascii=False)
        supported_product = case.get("supported_product", "")
        db.conn.execute(
            "INSERT INTO test_cases (task_id, operator_name, case_index, case_name, case_data, constraint_doc_id, supported_product) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, operator_name, idx, case_name, case_data, constraint_doc_id, supported_product),
        )
    db.conn.commit()
    logger.info("Saved %d test cases for task %s", len(cases), task_id)
    return {"saved_count": len(cases), "task_id": task_id}


def query_test_cases(
    task_id: str | None = None,
    operator_name: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Query test cases by task_id or operator_name."""
    db = get_db()
    conditions = []
    params: list = []
    if task_id:
        conditions.append("task_id = ?")
        params.append(task_id)
    if operator_name:
        conditions.append("operator_name = ?")
        params.append(operator_name)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    rows = db.conn.execute(
        f"SELECT id, task_id, operator_name, case_index, case_name, case_data, "
        f"constraint_doc_id, supported_product, created_at FROM test_cases{where} "
        f"ORDER BY case_index LIMIT ?",
        params,
    ).fetchall()
    cols = ["id", "task_id", "operator_name", "case_index", "case_name",
            "case_data", "constraint_doc_id", "supported_product", "created_at"]
    results = []
    for r in rows:
        row = dict(zip(cols, r, strict=False))
        try:
            row["case_data"] = json.loads(row["case_data"])
        except (json.JSONDecodeError, TypeError):
            pass
        results.append(row)
    return results


def get_latest_cases_task_id(operator_name: str) -> str | None:
    """Find the latest completed case_generate task for an operator."""
    return find_parent_task(operator_name, "case_generate")


# ── Exec Results ────────────────────────────────────────────────────────────

def save_exec_results(
    task_id: str,
    operator_name: str,
    results: list[dict],
) -> dict:
    """Save execution results. Each result dict must have case_id and passed."""
    db = get_db()
    count = 0
    for r in results:
        db.conn.execute(
            "INSERT INTO exec_results "
            "(task_id, case_id, operator_name, passed, cpu_precision_passed, "
            "precision_detail, actual_json, error_message, cpu_reference_code, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                r["case_id"],
                operator_name,
                r.get("passed", 0),
                r.get("cpu_precision_passed"),
                r.get("precision_detail"),
                json.dumps(r.get("actual"), ensure_ascii=False) if r.get("actual") else None,
                r.get("error_message"),
                r.get("cpu_reference_code"),
                r.get("duration_ms"),
            ),
        )
        count += 1
    db.conn.commit()
    logger.info("Saved %d exec results for task %s", count, task_id)
    return {"saved_count": count, "task_id": task_id}


def query_exec_results(
    task_id: str | None = None,
    case_id: int | None = None,
    operator_name: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Query exec results by task_id, case_id, or operator_name."""
    db = get_db()
    conditions = []
    params: list = []
    if task_id:
        conditions.append("er.task_id = ?")
        params.append(task_id)
    if case_id:
        conditions.append("er.case_id = ?")
        params.append(case_id)
    if operator_name:
        conditions.append("er.operator_name = ?")
        params.append(operator_name)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    rows = db.conn.execute(
        f"SELECT er.id, er.task_id, er.case_id, er.operator_name, er.passed, "
        f"er.cpu_precision_passed, er.precision_detail, er.actual_json, "
        f"er.error_message, er.cpu_reference_code, er.duration_ms, er.created_at, "
        f"tc.case_name, tc.case_data "
        f"FROM exec_results er "
        f"JOIN test_cases tc ON er.case_id = tc.id"
        f"{where} "
        f"ORDER BY er.id LIMIT ?",
        params,
    ).fetchall()
    cols = ["id", "task_id", "case_id", "operator_name", "passed",
            "cpu_precision_passed", "precision_detail", "actual_json",
            "error_message", "cpu_reference_code", "duration_ms", "created_at",
            "case_name", "case_data"]
    results = []
    for r in rows:
        row = dict(zip(cols, r, strict=False))
        for key in ("actual_json", "case_data"):
            try:
                row[key] = json.loads(row[key]) if row[key] else None
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(row)
    return results


# ── Servers ─────────────────────────────────────────────────────────────────

def create_server(name: str, ip: str, port: int, username: str, password: str, supported_product: str = "") -> int:
    """Create a new server record. Returns the new server id."""
    db = get_db()
    cursor = db.conn.execute(
        "INSERT INTO servers (name, ip, port, username, password, supported_product) VALUES (?, ?, ?, ?, ?, ?)",
        (name, ip, port, username, password, supported_product),
    )
    db.conn.commit()
    return cursor.lastrowid


def update_server(server_id: int, **fields) -> bool:
    """Update a server. Only provided fields are updated."""
    if not fields:
        return False
    allowed = {"name", "ip", "port", "username", "password", "status", "supported_product"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    set_clause += ", updated_at = datetime('now')"
    values = list(updates.values()) + [server_id]
    db = get_db()
    db.conn.execute(f"UPDATE servers SET {set_clause} WHERE id = ?", values)
    db.conn.commit()
    return True


def delete_server(server_id: int) -> bool:
    """Delete a server by id."""
    db = get_db()
    cursor = db.conn.execute("DELETE FROM servers WHERE id = ?", (server_id,))
    db.conn.commit()
    return cursor.rowcount > 0


def query_servers() -> list[dict]:
    """Query all servers, ordered by created_at DESC (newest first)."""
    db = get_db()
    rows = db.conn.execute(
        "SELECT id, name, ip, port, username, password, supported_product, status, created_at, updated_at "
        "FROM servers ORDER BY id DESC"
    ).fetchall()
    cols = ["id", "name", "ip", "port", "username", "password", "supported_product", "status", "created_at", "updated_at"]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def get_server(server_id: int) -> dict | None:
    """Get a single server by id."""
    db = get_db()
    row = db.conn.execute(
        "SELECT id, name, ip, port, username, password, supported_product, status, created_at, updated_at "
        "FROM servers WHERE id = ?",
        (server_id,),
    ).fetchone()
    if not row:
        return None
    cols = ["id", "name", "ip", "port", "username", "password", "supported_product", "status", "created_at", "updated_at"]
    return dict(zip(cols, row, strict=False))
