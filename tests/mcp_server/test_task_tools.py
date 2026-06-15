"""Tests for mcp_server task_tools — reset_stuck_task_items."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mcp_server.db import Database, get_db
from mcp_server.tools.task_tools import (
    create_task,
    create_task_items,
    get_pending_task_items,
    get_task,
    refresh_task_progress,
    reset_stuck_task_items,
    update_task_item_status,
    update_task_status,
)


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Create a fresh in-memory-like SQLite DB for each test."""
    db_path = tmp_path / "test.db"
    database = Database(str(db_path))
    database.connect()
    # Monkey-patch get_db to return this instance
    import mcp_server.db as db_mod
    db_mod._db = database
    yield database
    database.close()


@pytest.fixture()
def task_with_items(db: Database) -> int:
    """Create a task with 5 items and return the task_id."""
    result = create_task("test-batch", 5, "uploads/test")
    task_id = result["task_id"]
    items = [
        {"seq": i, "operator_name": f"op{i}", "file_path": f"uploads/test/op{i}.md"}
        for i in range(1, 6)
    ]
    create_task_items(task_id, items)
    return task_id


class TestResetStuckTaskItems:
    """Test reset_stuck_task_items function."""

    def test_no_stuck_items(self, db: Database, task_with_items: int) -> None:
        """When no items are stuck in 'running', reset_count should be 0."""
        result = reset_stuck_task_items(task_with_items)
        assert result["reset_count"] == 0

    def test_resets_running_items(self, db: Database, task_with_items: int) -> None:
        """Items stuck in 'running' should be reset to 'pending'."""
        # Simulate item 3 stuck in 'running'
        update_task_item_status(3, "running", started_at="2026-01-01T00:00:00Z")
        update_task_status(task_with_items, "running")

        result = reset_stuck_task_items(task_with_items)
        assert result["reset_count"] == 1

        # Verify the item is back to pending
        pending = get_pending_task_items(task_with_items)
        pending_ids = [p["id"] for p in pending]
        assert 3 in pending_ids

    def test_resets_multiple_running_items(
        self, db: Database, task_with_items: int
    ) -> None:
        """Multiple stuck items should all be reset."""
        # Items 2 and 4 stuck in running
        update_task_item_status(2, "running", started_at="2026-01-01T00:00:00Z")
        update_task_item_status(4, "running", started_at="2026-01-01T00:00:00Z")

        result = reset_stuck_task_items(task_with_items)
        assert result["reset_count"] == 2

    def test_preserves_completed_and_failed(
        self, db: Database, task_with_items: int
    ) -> None:
        """Completed and failed items should not be affected."""
        update_task_item_status(1, "completed", doc_id=100)
        update_task_item_status(2, "failed", error="test error")
        update_task_item_status(3, "running", started_at="2026-01-01T00:00:00Z")

        result = reset_stuck_task_items(task_with_items)
        assert result["reset_count"] == 1

        # Items 4 and 5 should still be pending (along with 3 now)
        pending = get_pending_task_items(task_with_items)
        pending_ids = [p["id"] for p in pending]
        assert 3 in pending_ids
        assert 4 in pending_ids
        assert 5 in pending_ids
        # Completed and failed items should NOT be pending
        assert 1 not in pending_ids
        assert 2 not in pending_ids

    def test_resets_task_status_from_running(
        self, db: Database, task_with_items: int
    ) -> None:
        """Task status should be reset from 'running' to 'pending'."""
        update_task_status(task_with_items, "running")
        task = get_task(task_with_items)
        assert task["status"] == "running"

        reset_stuck_task_items(task_with_items)

        task = get_task(task_with_items)
        assert task["status"] == "pending"

    def test_does_not_reset_completed_task(
        self, db: Database, task_with_items: int
    ) -> None:
        """Task with 'completed' status should not be changed."""
        update_task_status(task_with_items, "completed")

        reset_stuck_task_items(task_with_items)

        task = get_task(task_with_items)
        assert task["status"] == "completed"

    def test_clears_started_at_on_reset(
        self, db: Database, task_with_items: int
    ) -> None:
        """started_at should be cleared when item is reset to pending."""
        update_task_item_status(3, "running", started_at="2026-01-01T12:00:00Z")

        reset_stuck_task_items(task_with_items)

        # Query the item directly to check started_at is NULL
        conn = db.conn
        row = conn.execute(
            "SELECT status, started_at FROM task_items WHERE id = 3"
        ).fetchone()
        assert row[0] == "pending"
        assert row[1] is None

    def test_pending_items_remain_after_reset(
        self, db: Database, task_with_items: int
    ) -> None:
        """Already-pending items should still be pending after reset."""
        # Only mark item 1 as completed, rest stay pending
        update_task_item_status(1, "completed", doc_id=100)
        # Mark item 3 as running (stuck)
        update_task_item_status(3, "running", started_at="2026-01-01T00:00:00Z")

        reset_stuck_task_items(task_with_items)

        # Items 2, 3, 4, 5 should all be pending
        pending = get_pending_task_items(task_with_items)
        assert len(pending) == 4
