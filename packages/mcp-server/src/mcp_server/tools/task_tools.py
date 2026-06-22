"""MCP Tools for task management: create, update, and query batch processing tasks."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mcp_server.db import get_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_task(name: str, total_count: int, upload_dir: str) -> dict:
    """Create a new task record.

    Args:
        name: Task name.
        total_count: Total number of documents.
        upload_dir: Upload directory path.

    Returns:
        dict with task_id, status.
    """
    db = get_db()
    conn = db.conn
    now = _now_iso()
    cursor = conn.execute(
        "INSERT INTO tasks (name, status, total_count, upload_dir, created_at, updated_at) "
        "VALUES (?, 'pending', ?, ?, ?, ?)",
        (name, total_count, upload_dir, now, now),
    )
    conn.commit()
    return {"task_id": cursor.lastrowid, "status": "pending"}


def create_task_items(task_id: int, items: list[dict]) -> dict:
    """Batch insert task_items.

    Args:
        task_id: Parent task ID.
        items: List of dicts with seq, operator_name, file_path.

    Returns:
        dict with count of inserted items.
    """
    db = get_db()
    conn = db.conn
    for item in items:
        conn.execute(
            "INSERT INTO task_items (task_id, seq, operator_name, file_path) "
            "VALUES (?, ?, ?, ?)",
            (task_id, item["seq"], item["operator_name"], item["file_path"]),
        )
    conn.commit()
    return {"count": len(items)}


def update_task_status(task_id: int, status: str) -> dict:
    """Update task status and updated_at timestamp.

    Args:
        task_id: Task ID.
        status: New status (pending/running/completed/failed).

    Returns:
        dict with updated flag.
    """
    db = get_db()
    conn = db.conn
    now = _now_iso()
    conn.execute(
        "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
        (status, now, task_id),
    )
    conn.commit()
    return {"updated": True}


def update_task_item_status(
    item_id: int,
    status: str,
    error: str | None = None,
    doc_id: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict:
    """Update task item status and optional fields.

    Args:
        item_id: Task item ID.
        status: New status.
        error: Error message (if failed).
        doc_id: Document version ID (if completed).
        started_at: Start timestamp.
        finished_at: Finish timestamp.

    Returns:
        dict with updated flag.
    """
    db = get_db()
    conn = db.conn
    updates = ["status = ?"]
    values: list = [status]
    if error is not None:
        updates.append("error = ?")
        values.append(error)
    if doc_id is not None:
        updates.append("doc_id = ?")
        values.append(doc_id)
    if started_at is not None:
        updates.append("started_at = ?")
        values.append(started_at)
    if finished_at is not None:
        updates.append("finished_at = ?")
        values.append(finished_at)
    values.append(item_id)
    conn.execute(
        f"UPDATE task_items SET {', '.join(updates)} WHERE id = ?",
        values,
    )
    conn.commit()
    return {"updated": True}


def get_pending_task_items(task_id: int) -> list[dict]:
    """Get all pending task items for a task, ordered by seq.

    Args:
        task_id: Task ID.

    Returns:
        List of task item dicts.
    """
    db = get_db()
    conn = db.conn
    rows = conn.execute(
        "SELECT id, task_id, seq, operator_name, file_path, status "
        "FROM task_items WHERE task_id = ? AND status = 'pending' ORDER BY seq",
        (task_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "task_id": r[1],
            "seq": r[2],
            "operator_name": r[3],
            "file_path": r[4],
            "status": r[5],
        }
        for r in rows
    ]


def get_task(task_id: int) -> dict | None:
    """Get a single task by ID.

    Args:
        task_id: Task ID.

    Returns:
        Task dict or None.
    """
    db = get_db()
    conn = db.conn
    row = conn.execute(
        "SELECT id, name, status, total_count, completed_count, failed_count, "
        "upload_dir, created_at, updated_at FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "status": row[2],
        "total_count": row[3],
        "completed_count": row[4],
        "failed_count": row[5],
        "upload_dir": row[6],
        "created_at": row[7],
        "updated_at": row[8],
    }


def list_tasks() -> list[dict]:
    """List all tasks ordered by created_at DESC.

    Returns:
        List of task dicts.
    """
    db = get_db()
    conn = db.conn
    rows = conn.execute(
        "SELECT id, name, status, total_count, completed_count, failed_count, "
        "upload_dir, created_at, updated_at FROM tasks ORDER BY created_at DESC"
    ).fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "status": r[2],
            "total_count": r[3],
            "completed_count": r[4],
            "failed_count": r[5],
            "upload_dir": r[6],
            "created_at": r[7],
            "updated_at": r[8],
        }
        for r in rows
    ]


def get_task_with_items(task_id: int) -> dict | None:
    """Get a task with all its items.

    Args:
        task_id: Task ID.

    Returns:
        Task dict with items list, or None.
    """
    db = get_db()
    conn = db.conn
    task = get_task(task_id)
    if not task:
        return None
    rows = conn.execute(
        "SELECT id, seq, operator_name, file_path, status, doc_id, error, "
        "started_at, finished_at FROM task_items WHERE task_id = ? ORDER BY seq",
        (task_id,),
    ).fetchall()
    task["items"] = [
        {
            "id": r[0],
            "seq": r[1],
            "operator_name": r[2],
            "file_path": r[3],
            "status": r[4],
            "doc_id": r[5],
            "error": r[6],
            "started_at": r[7],
            "finished_at": r[8],
        }
        for r in rows
    ]
    return task


def refresh_task_progress(task_id: int) -> dict:
    """Recount completed/failed items and update task progress.

    Args:
        task_id: Task ID.

    Returns:
        dict with updated counts.
    """
    db = get_db()
    conn = db.conn
    completed = conn.execute(
        "SELECT COUNT(*) FROM task_items WHERE task_id = ? AND status = 'completed'",
        (task_id,),
    ).fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(*) FROM task_items WHERE task_id = ? AND status = 'failed'",
        (task_id,),
    ).fetchone()[0]
    now = _now_iso()
    conn.execute(
        "UPDATE tasks SET completed_count = ?, failed_count = ?, updated_at = ? WHERE id = ?",
        (completed, failed, now, task_id),
    )
    conn.commit()
    return {"completed_count": completed, "failed_count": failed}


def reset_stuck_task_items(task_id: int) -> dict:
    """Reset task items stuck in 'running' back to 'pending'.

    This handles the case where the server crashed or was restarted while
    items were being processed.  Those items remain in 'running' status
    indefinitely and block subsequent execution.

    Also resets the parent task status to 'pending' if it was 'running'.

    Args:
        task_id: Task ID.

    Returns:
        dict with count of reset items.
    """
    db = get_db()
    conn = db.conn

    # Count stuck items first
    stuck_count = conn.execute(
        "SELECT COUNT(*) FROM task_items WHERE task_id = ? AND status = 'running'",
        (task_id,),
    ).fetchone()[0]

    if stuck_count > 0:
        # Reset stuck items to pending, clear started_at
        conn.execute(
            "UPDATE task_items SET status = 'pending', started_at = NULL "
            "WHERE task_id = ? AND status = 'running'",
            (task_id,),
        )

    # Reset task status from 'running' to 'pending'
    conn.execute(
        "UPDATE tasks SET status = 'pending', updated_at = ? "
        "WHERE id = ? AND status = 'running'",
        (_now_iso(), task_id),
    )
    conn.commit()

    return {"reset_count": stuck_count}


# ---------------------------------------------------------------------------
# Tables that reference document_versions(doc_id) — must be cleaned up
# before deleting the document_versions row itself.
# ---------------------------------------------------------------------------
_DOC_ID_CHILD_TABLES = [
    "parameters",
    "param_relations",
    "function_signatures",
    "platform_support",
    "return_codes",
    "dtype_combinations",
    "constraints_result",
    "shape_dim_mappings",
    "platform_constants",
]


def delete_task(task_id: int) -> dict:
    """Delete a task and all associated operator data.

    Cascade deletion order:
    1. Collect all doc_ids from task_items
    2. For each doc_id: delete from all child tables, then document_versions
    3. Delete task_items
    4. Delete the task record

    Only allows deletion of finished tasks (completed, failed, pending).
    Refuses to delete running tasks.

    Args:
        task_id: Task ID.

    Returns:
        dict with deleted_task_id, deleted_docs count, deleted_items count.

    Raises:
        ValueError: If task not found or still running.
    """
    db = get_db()
    conn = db.conn

    # Verify task exists and is not running
    task = conn.execute(
        "SELECT id, status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if task is None:
        raise ValueError(f"Task {task_id} not found")
    if task[1] == "running":
        raise ValueError("Cannot delete a task that is still running")

    # Collect doc_ids from task items (only completed items have doc_id)
    rows = conn.execute(
        "SELECT doc_id FROM task_items WHERE task_id = ? AND doc_id IS NOT NULL",
        (task_id,),
    ).fetchall()
    doc_ids = [r[0] for r in rows if r[0]]

    # Delete child table rows for each doc_id
    deleted_docs = 0
    for doc_id in doc_ids:
        for table in _DOC_ID_CHILD_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM document_versions WHERE id = ?", (doc_id,))
        deleted_docs += 1

    # Delete task items
    conn.execute("DELETE FROM task_items WHERE task_id = ?", (task_id,))

    # Delete the task record
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    conn.commit()

    return {
        "deleted_task_id": task_id,
        "deleted_docs": deleted_docs,
        "deleted_items": len(rows),
    }
