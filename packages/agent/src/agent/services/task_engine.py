"""Background task execution engine for batch operator document processing."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from agent.graph import create_pipeline_graph
from agent.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# Global lock: only one task runs at a time to avoid LLM concurrency limits
_run_lock = asyncio.Lock()


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


async def run_task(task_id: int) -> None:
    """Execute all pending items in a task sequentially.

    Acquires a global lock to ensure only one task runs at a time.
    For each pending item:
    1. Set item status to 'running'
    2. Read file content and compute hash
    3. Invoke the pipeline graph
    4. Set item status to 'completed' or 'failed'
    5. Refresh task progress
    After all items, set task status to 'completed' or 'failed'.
    """
    async with _run_lock:
        mcp = MCPClient()
        graph = create_pipeline_graph()

        # Set task to running
        await mcp.update_task_status(task_id, "running")

        # Get all pending items
        items = await mcp.get_pending_task_items(task_id)

        for item in items:
            item_id = item["id"]
            file_path = item["file_path"]
            operator_name = item["operator_name"]

            # Set item to running
            await mcp.update_task_item_status(
                item_id, "running", started_at=_now_iso()
            )

            try:
                # Read file content
                content = Path(file_path).read_text(encoding="utf-8")
                content_hash = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()

                # Execute pipeline
                result = await graph.ainvoke(
                    {
                        "operator_name": operator_name,
                        "content": content,
                        "content_hash": content_hash,
                    }
                )

                pipeline_error = result.get("error")
                if pipeline_error:
                    await mcp.update_task_item_status(
                        item_id,
                        "failed",
                        error=pipeline_error,
                        doc_id=result.get("doc_id"),
                        finished_at=_now_iso(),
                    )
                else:
                    await mcp.update_task_item_status(
                        item_id,
                        "completed",
                        doc_id=result.get("doc_id"),
                        finished_at=_now_iso(),
                    )

            except Exception as e:
                logger.exception("Task item %s failed: %s", item_id, e)
                await mcp.update_task_item_status(
                    item_id,
                    "failed",
                    error=str(e),
                    finished_at=_now_iso(),
                )

            # Refresh task progress
            await mcp.refresh_task_progress(task_id)

        # Set final task status
        task = await mcp.get_task(task_id)
        if task is None:
            logger.error("Task %s not found after execution", task_id)
            return

        final_status = "completed" if task["failed_count"] == 0 else "failed"
        await mcp.update_task_status(task_id, final_status)
        logger.info(
            "Task %s finished: %s (completed=%d, failed=%d)",
            task_id,
            final_status,
            task["completed_count"],
            task["failed_count"],
        )


async def retry_failed_task(task_id: int) -> dict:
    """Retry failed operators from a completed task.

    Extracts all failed items from the original task, creates a new
    task containing only those items, and kicks off background execution.

    Args:
        task_id: The original task ID whose failed items should be retried.

    Returns:
        dict with new_task_id, new_task_name, failed_count, upload_dir.

    Raises:
        ValueError: If the task is not found, still running, or has no
            failed items.
    """
    mcp = MCPClient()

    # Verify the original task exists
    task = await mcp.get_task(task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")

    # Only allow retry for finished tasks (completed or failed)
    if task["status"] == "running":
        raise ValueError("Cannot retry a task that is still running")

    # Fetch all items and filter failed ones
    task_detail = await mcp.get_task_with_items(task_id)
    if task_detail is None:
        raise ValueError(f"Task {task_id} detail not found")

    failed_items = [
        item for item in task_detail.get("items", [])
        if item["status"] == "failed"
    ]
    if not failed_items:
        raise ValueError("No failed items to retry")

    # Build new task name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = task["name"]
    new_name = f"{original_name}-retry-{timestamp}"

    # Reuse the original upload_dir (files are still on disk)
    upload_dir = task["upload_dir"]

    # Build item list for the new task
    items = [
        {
            "seq": seq,
            "operator_name": item["operator_name"],
            "file_path": item["file_path"],
        }
        for seq, item in enumerate(failed_items, start=1)
    ]

    # Create new task + items in DB
    result = await mcp.create_task(new_name, len(items), upload_dir)
    new_task_id = result["task_id"]
    await mcp.create_task_items(new_task_id, items)

    # Kick off background execution
    asyncio.create_task(run_task(new_task_id))

    logger.info(
        "Created retry task %s (name=%s) for %d failed items from task %s",
        new_task_id,
        new_name,
        len(items),
        task_id,
    )

    return {
        "new_task_id": new_task_id,
        "new_task_name": new_name,
        "failed_count": len(items),
        "upload_dir": upload_dir,
    }


async def resume_task(task_id: int) -> dict:
    """Resume a stuck task by resetting stuck items and re-running.

    1. Reset all task items stuck in 'running' back to 'pending'.
    2. Re-run the task via run_task() which picks up pending items.

    Returns:
        dict with reset_count (number of items reset from 'running').
    """
    mcp = MCPClient()

    # Reset stuck items
    result = await mcp.reset_stuck_task_items(task_id)
    reset_count = result.get("reset_count", 0)

    logger.info(
        "Resuming task %s: reset %d stuck items to pending",
        task_id,
        reset_count,
    )

    # Re-run the task (acquires lock internally)
    asyncio.create_task(run_task(task_id))

    return {"reset_count": reset_count, "task_id": task_id}
