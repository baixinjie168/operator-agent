"""Background task execution engine for batch operator document processing."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from agent.core.config import settings
from agent.graph import PipelineStage, build_pipeline
from agent.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# Global lock: only one task runs at a time to avoid LLM concurrency limits
_run_lock = asyncio.Lock()

# Cancellation registry: task_id -> asyncio.Event (cooperative cancel flag)
_cancel_flags: dict[int, asyncio.Event] = {}
# Running task handles: task_id -> asyncio.Task (for hard cancellation)
_task_handles: dict[int, asyncio.Task] = {}


def _is_cancelled(task_id: int) -> bool:
    """Check whether a task has been flagged for cancellation."""
    evt = _cancel_flags.get(task_id)
    return evt is not None and evt.is_set()


from shared.utils import now_iso as _now_iso


async def _process_item(
    item: dict,
    graph,
    mcp: MCPClient,
) -> None:
    """Process a single task item: read file, run pipeline, update status.

    This function is designed to be called concurrently via asyncio.gather
    with a Semaphore controlling the maximum parallelism.
    """
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


async def _get_failed_items(
    mcp: MCPClient, task_id: int,
) -> list[dict]:
    """Fetch failed items from a task after a processing pass."""
    task_detail = await mcp.get_task_with_items(task_id)
    if task_detail is None:
        return []
    return [
        item for item in task_detail.get("items", [])
        if item["status"] == "failed"
    ]


async def _reset_items_to_pending(
    mcp: MCPClient, items: list[dict],
) -> None:
    """Reset failed items back to 'pending' status for retry."""
    for item in items:
        await mcp.update_task_item_status(item["id"], "pending")


async def _run_constraint_checks(task_id: int, mcp: MCPClient) -> None:
    """Run constraint check for all successfully parsed operators in a task.

    Called automatically after task completion. Fully isolated:
    - Each operator is checked in its own try/except
    - Failure for one operator does not affect others
    - Overall failure does not affect task status
    """
    from agent.nodes.constraint_check_agent import run_constraint_check

    items = await mcp.get_task_items(task_id)
    completed_items = [
        it for it in items
        if it.get("status") == "completed" and it.get("doc_id")
    ]

    if not completed_items:
        return

    logger.info(
        "Task %s: starting constraint check for %d operators",
        task_id, len(completed_items),
    )

    checked = 0
    for item in completed_items:
        if _is_cancelled(task_id):
            break
        try:
            doc = await mcp.get_doc_for_check(item["doc_id"])
            if not doc:
                continue
            json_constraints = doc.get("json_constraints", "{}")
            content = doc.get("content", "")
            operator_name = doc.get("operator_name") or item.get("operator_name", "")
            if not content.strip() or json_constraints == "{}":
                continue

            html = await run_constraint_check(
                content, json_constraints, operator_name,
            )
            if html:
                await mcp.save_constraint_check_report(item["doc_id"], html)
                checked += 1
                logger.info(
                    "Task %s: checked %s (%d/%d)",
                    task_id, item["operator_name"],
                    checked, len(completed_items),
                )
        except Exception:
            logger.warning(
                "Task %s: constraint check failed for %s, skipping",
                task_id, item.get("operator_name", "?"),
                exc_info=True,
            )

    logger.info(
        "Task %s: constraint check done: %d/%d operators",
        task_id, checked, len(completed_items),
    )


async def run_task(task_id: int, max_workers: int | None = None) -> None:
    """Execute all pending items in a task with controlled parallelism.

    Acquires a global lock to ensure only one task runs at a time.
    Within the task, items are processed in parallel using a Semaphore
    (controlled by max_workers or settings.task_max_workers, default 3).

    After the first pass, failed items are automatically retried up to
    settings.task_max_retries times (default 1).

    Supports cooperative cancellation via ``stop_task``: a cancellation
    flag is checked before each item and between retry rounds.  If the
    flag is set, remaining items are skipped and the task is marked
    'cancelled'.

    Args:
        task_id: The task to execute.
        max_workers: Override for parallelism (None = use settings.task_max_workers).
    """
    # Register cancellation flag and current task handle
    cancel_event = asyncio.Event()
    _cancel_flags[task_id] = cancel_event
    current = asyncio.current_task()
    if current is not None:
        _task_handles[task_id] = current

    try:
        async with _run_lock:
            mcp = MCPClient()
            graph = build_pipeline([PipelineStage.EXTRACT])
            workers = max_workers if max_workers is not None else settings.task_max_workers
            max_retries = settings.task_max_retries

            # Set task to running
            await mcp.update_task_status(task_id, "running")

            # Get all pending items
            items = await mcp.get_pending_task_items(task_id)

            logger.info(
                "Task %s: processing %d items with max_workers=%d, max_retries=%d",
                task_id, len(items), workers, max_retries,
            )

            # Process items in parallel with Semaphore-controlled concurrency
            sem = asyncio.Semaphore(workers)

            async def _process_with_sem(item: dict) -> None:
                async with sem:
                    # Cooperative cancellation: skip if task was stopped
                    if _is_cancelled(task_id):
                        return
                    await _process_item(item, graph, mcp)
                    # Refresh task progress after each item completes
                    await mcp.refresh_task_progress(task_id)

            await asyncio.gather(*[_process_with_sem(item) for item in items])

            # Retry loop: re-process failed items
            for retry_round in range(1, max_retries + 1):
                # Check cancellation before each retry round
                if _is_cancelled(task_id):
                    logger.info("Task %s: cancelled, skipping retry", task_id)
                    break

                failed_items = await _get_failed_items(mcp, task_id)
                if not failed_items:
                    logger.info(
                        "Task %s: no failed items after pass, skipping retry",
                        task_id,
                    )
                    break

                logger.info(
                    "Task %s: retry round %d/%d — %d failed items",
                    task_id, retry_round, max_retries, len(failed_items),
                )

                # Reset failed items to pending for re-processing
                await _reset_items_to_pending(mcp, failed_items)

                # Re-process failed items in parallel
                await asyncio.gather(
                    *[_process_with_sem(item) for item in failed_items]
                )

            # Set final task status
            task = await mcp.get_task(task_id)
            if task is None:
                logger.error("Task %s not found after execution", task_id)
                return

            if _is_cancelled(task_id):
                # Cooperative cancellation path: clean up and mark cancelled
                await mcp.reset_stuck_task_items(task_id)
                await mcp.update_task_status(task_id, "cancelled")
                logger.info("Task %s cancelled by user", task_id)
            else:
                final_status = "completed" if task["failed_count"] == 0 else "failed"
                await mcp.update_task_status(task_id, final_status)
                logger.info(
                    "Task %s finished: %s (completed=%d, failed=%d)",
                    task_id,
                    final_status,
                    task["completed_count"],
                    task["failed_count"],
                )

            # --- New: automatic constraint check after task completion ---
            if not _is_cancelled(task_id):
                try:
                    await _run_constraint_checks(task_id, mcp)
                except Exception:
                    logger.warning(
                        "Task %s: constraint check phase failed, skipping",
                        task_id, exc_info=True,
                    )
            # --- End constraint check ---
    finally:
        # Always clean up the registries, even if cancelled via CancelledError
        _cancel_flags.pop(task_id, None)
        _task_handles.pop(task_id, None)


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

    # Kick off background execution with concurrency=1 for retry safety
    asyncio.create_task(run_task(new_task_id, max_workers=1))

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
    from agent.routes.task import _get_max_workers
    asyncio.create_task(run_task(task_id, max_workers=_get_max_workers()))

    return {"reset_count": reset_count, "task_id": task_id}


async def retry_item(task_id: int, item_id: int) -> dict:
    """Retry a single failed task item.

    Resets the item to 'pending' (clearing error and timestamps), then
    re-runs the task.  ``run_task`` picks up all pending items — typically
    just the one we reset.

    Args:
        task_id: The task that owns the item.
        item_id: The specific task item to retry.

    Returns:
        dict with task_id and item_id.

    Raises:
        ValueError: If the task is not found or still running.
    """
    mcp = MCPClient()

    task = await mcp.get_task(task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")
    if task["status"] == "running":
        raise ValueError("Cannot retry an item while the task is still running")

    # Reset the specific item to pending
    await mcp.reset_task_item(item_id)

    # Set task status to pending so run_task can pick it up
    await mcp.update_task_status(task_id, "pending")

    logger.info("Retrying item %s in task %s", item_id, task_id)

    # Re-run the task (acquires lock internally)
    from agent.routes.task import _get_max_workers
    asyncio.create_task(run_task(task_id, max_workers=_get_max_workers()))

    return {"task_id": task_id, "item_id": item_id}


async def stop_task(task_id: int) -> dict:
    """Stop a running task (called from the HTTP route layer).

    Uses a dual cancellation strategy:

    1. **Cooperative**: set an ``asyncio.Event`` flag so that
       ``run_task`` skips items that have not yet started processing.
    2. **Interruptive**: ``cancel()`` the ``asyncio.Task`` handle to
       interrupt in-flight LLM calls.  ``CancelledError`` is a
       ``BaseException`` and is *not* caught by ``_process_item``'s
       ``except Exception``, so items may be left in 'running'.
    3. **Database cleanup**: after the task handle exits, call the MCP
       ``stop_task`` tool to reset any 'running' items to 'pending' and
       set the task status to 'cancelled'.

    Args:
        task_id: The task to stop.

    Returns:
        dict with task_id and reset_count.

    Raises:
        ValueError: If the task is not found or is not running.
    """
    mcp = MCPClient()

    # Verify task exists and is running
    task = await mcp.get_task(task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")
    if task["status"] != "running":
        raise ValueError(
            f"Task {task_id} is not running (status={task['status']})"
        )

    # 1. Set cooperative cancellation flag
    evt = _cancel_flags.get(task_id)
    if evt:
        evt.set()

    # 2. Interruptive: cancel the asyncio task handle
    handle = _task_handles.get(task_id)
    if handle:
        handle.cancel()
        try:
            await handle
        except (asyncio.CancelledError, Exception):
            pass

    # 3. Database-side cleanup (reset running items, set status to cancelled)
    result = await mcp.stop_task(task_id)

    logger.info(
        "Stopped task %s: reset %d running items",
        task_id, result.get("reset_count", 0),
    )

    return {
        "task_id": task_id,
        "reset_count": result.get("reset_count", 0),
    }

