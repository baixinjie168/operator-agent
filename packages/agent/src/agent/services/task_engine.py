"""Background task execution engine for batch operator document processing."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from agent.core.config import settings
from agent.db import complete_run as db_complete_run
from agent.db import create_run as db_create_run
from agent.db import save_events as db_save_events
from agent.db import update_run_doc_id as db_update_run_doc_id
from agent.graph import PipelineStage, build_pipeline
from agent.mcp_client import MCPClient
from agent.nodes.init_doc import init_doc_node as _init_doc
from agent.runtime import EventType, LLMTracer, RuntimeManager, traced_node

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


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _serialize_events(run) -> list[dict]:
    """Convert ``RuntimeRun.events`` to the dict shape ``db.save_events`` expects.

    Uses the alias names emitted by ``to_sse()`` (e.g. ``"node.started"``)
    because the frontend's ``_eventRouteMap`` only recognises those — saving
    raw enum values would silently break replays after a backend restart.
    """
    payload: list[dict] = []
    for evt in run.events:
        sse = evt.to_sse()
        payload.append({
            "seq": evt.seq,
            "event_type": sse["event_type"],
            "data": sse["data"],
        })
    return payload


async def _process_item(
    item: dict,
    graph,
    mcp: MCPClient,
    task_id: int,
    run_id: str,
    manager: RuntimeManager,
) -> None:
    """Process a single task item with the batch-mirror pipeline.

    Pipeline (matches the single-upload flow in ``routes/upload.py``):
    1. ``init_doc`` — creates the ``document_versions`` row and sets
       ``state["doc_id"]`` so downstream nodes (``function_explanation_extract``,
       ``assemble_result``) have a doc to write to.
    2. EXTRACT graph — produces the constraint payload.
    3. ``manager.emit(WORKFLOW_END, ...)`` — drives the frontend's
       ``taskCompleted`` handler that renders the MasterAgent summary
       + "是否继续生成测试用例？" prompt.
    4. Persist events to ``pipeline_events`` so clicking the task replays
       the full timeline including the WORKFLOW_END.

    Designed to be called concurrently via ``asyncio.gather`` with a
    Semaphore controlling the maximum parallelism.
    """
    item_id = item["id"]
    file_path = item["file_path"]
    operator_name = item["operator_name"]

    run = manager.get_run(run_id)
    if run is None:
        logger.error("RuntimeRun %s not found in manager; aborting item %s",
                     run_id, item_id)
        await mcp.update_task_item_status(
            item_id, "failed", error="internal: RuntimeRun not registered",
            finished_at=_now_iso(),
        )
        return

    # Bind the runtime context so traced nodes emit events into THIS run's
    # event list (not some other run's). Each coroutine in the gather
    # carries its own contextvar copy, so concurrent items do not collide.
    manager.enter_context(run_id)
    llm_tracer = LLMTracer()

    # Emit WORKFLOW_START so the frontend's eventRouter can pick up the
    # "task running" state and the architecture card can animate.
    manager.emit(EventType.WORKFLOW_START, run_id, run.spans[run_id], {
        "agent_id": "doc",
        "node_id": "init_doc",
        "message": f"DocAgent 开始处理 {operator_name}...",
        "step_index": 0, "progress_pct": 0, "progress_text": "开始",
    })

    # Set MCP item to running
    await mcp.update_task_item_status(
        item_id, "running", started_at=_now_iso()
    )

    try:
        content = Path(file_path).read_text(encoding="utf-8")
        content_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        state_input: dict = {
            "operator_name": operator_name,
            "content": content,
            "content_hash": content_hash,
            "run_id": run_id,
        }

        # Step 1: init_doc — must run before EXTRACT so downstream nodes
        # (function_explanation_extract, assemble_result) see a doc_id.
        # Without this, the MasterAgent summary is never saved.
        traced_init_doc = traced_node("init_doc")(_init_doc)
        init_result = await traced_init_doc(state_input)
        if isinstance(init_result, dict):
            state_input.update(init_result)
        if init_result.get("status") == "error":
            init_error = init_result.get("error", "init_doc failed")
            logger.warning("init_doc failed for %s: %s", operator_name, init_error)
            try:
                await asyncio.to_thread(
                    db_save_events, run_id, _serialize_events(run)
                )
            except Exception:
                logger.exception("Failed to persist events for %s", run_id)
            await mcp.update_task_item_status(
                item_id, "failed", error=init_error, finished_at=_now_iso(),
            )
            try:
                await asyncio.to_thread(
                    db_complete_run, run_id, {}, error=init_error
                )
            except Exception:
                logger.exception("Failed to mirror init_doc failure for %s", run_id)
            manager.complete_run(run_id, error=init_error)
            return

        init_doc_id = state_input.get("doc_id")
        if init_doc_id is not None:
            try:
                await asyncio.to_thread(
                    db_update_run_doc_id, run_id, init_doc_id
                )
            except Exception:
                logger.exception(
                    "Failed to update doc_id on pipeline_runs for %s", run_id
                )

        # Step 2: EXTRACT pipeline
        result = await graph.ainvoke(
            state_input, config={"callbacks": [llm_tracer]}
        )

        pipeline_error = result.get("error")
        doc_id = result.get("doc_id") or init_doc_id

        # Step 3: emit WORKFLOW_END with a result dict the frontend's
        # taskCompleted handler reads to render the MasterAgent summary.
        sections_count = len(result.get("sections", []) or [])
        parameters_count = len(result.get("parameters", []) or [])
        product_count = len(result.get("product_support", []) or [])
        version = result.get("version")
        op_name = result.get("operator_name", operator_name)
        manager.emit(EventType.WORKFLOW_END, run_id, run.spans[run_id], {
            "agent_id": "doc",
            "node_id": "init_doc",
            "message": f"DocParserAgent 完成。状态={'completed' if not pipeline_error else 'failed'}, v{version}",
            "summary": f"提取完成。{sections_count} sections, {parameters_count} 参数, {product_count} 产品。",
            "progress_pct": 100, "progress_text": "完成",
            "result": {
                "status": "completed" if not pipeline_error else "failed",
                "version": version,
                "sections_count": sections_count,
                "parameters_count": parameters_count,
                "product_count": product_count,
                "doc_id": doc_id,
                "operator_name": op_name,
                "run_id": run_id,
            },
        })

        # Persist events to pipeline_events. Order matters: WORKFLOW_END
        # must be emitted before this so it's included in the saved events.
        try:
            await asyncio.to_thread(db_save_events, run_id, _serialize_events(run))
        except Exception:
            logger.exception("Failed to persist events for run %s", run_id)

        # Update pipeline_runs status.
        try:
            if doc_id is not None:
                await asyncio.to_thread(db_update_run_doc_id, run_id, doc_id)
            await asyncio.to_thread(
                db_complete_run,
                run_id,
                result if not pipeline_error else {},
                error=pipeline_error,
                doc_id=doc_id,
            )
        except Exception:
            logger.exception(
                "Failed to mirror item %s to pipeline_runs", run_id
            )

        manager.complete_run(run_id, error=pipeline_error)

        if pipeline_error:
            await mcp.update_task_item_status(
                item_id,
                "failed",
                error=pipeline_error,
                doc_id=doc_id,
                finished_at=_now_iso(),
            )
        else:
            await mcp.update_task_item_status(
                item_id,
                "completed",
                doc_id=doc_id,
                finished_at=_now_iso(),
            )

    except Exception as e:
        logger.exception("Task item %s failed: %s", item_id, e)
        # Emit WORKFLOW_ERROR so the frontend's taskFailed handler fires.
        try:
            manager.emit(EventType.WORKFLOW_ERROR, run_id, run.spans[run_id], {
                "agent_id": "doc",
                "error": str(e),
            })
        except Exception:
            logger.exception("Failed to emit WORKFLOW_ERROR for %s", run_id)
        try:
            await asyncio.to_thread(db_save_events, run_id, _serialize_events(run))
        except Exception:
            logger.exception("Failed to persist partial events for %s", run_id)
        try:
            await asyncio.to_thread(
                db_complete_run, run_id, {}, error=str(e)
            )
        except Exception:
            logger.exception(
                "Failed to mirror failure for %s to pipeline_runs", run_id
            )
        manager.complete_run(run_id, error=str(e))
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


async def _process_item_no_mirror(
    item: dict,
    graph,
    mcp: MCPClient,
    task_id: int,
) -> None:
    """Fallback path when no RuntimeRun / run_id is available for an item.

    Runs the pipeline (with the same MCP status updates as the mirror path)
    but does NOT write to ``pipeline_runs`` or ``pipeline_events``. Used
    by the resume path and any other code that calls ``run_task`` without
    going through ``create_task``.
    """
    item_id = item["id"]
    file_path = item["file_path"]
    operator_name = item["operator_name"]

    await mcp.update_task_item_status(
        item_id, "running", started_at=_now_iso()
    )
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        result = await graph.ainvoke({
            "operator_name": operator_name,
            "content": content,
            "content_hash": content_hash,
        })
        pipeline_error = result.get("error")
        if pipeline_error:
            await mcp.update_task_item_status(
                item_id, "failed",
                error=pipeline_error, doc_id=result.get("doc_id"),
                finished_at=_now_iso(),
            )
        else:
            await mcp.update_task_item_status(
                item_id, "completed",
                doc_id=result.get("doc_id"), finished_at=_now_iso(),
            )
    except Exception as e:
        logger.exception("Task item %s (no-mirror) failed: %s", item_id, e)
        await mcp.update_task_item_status(
            item_id, "failed", error=str(e), finished_at=_now_iso(),
        )


async def run_task(
    task_id: int,
    max_workers: int | None = None,
    *,
    manager: RuntimeManager | None = None,
    batch_run_ids: list[dict] | None = None,
) -> None:
    """Execute all pending items in a task with controlled parallelism.

    Acquires a global lock to ensure only one task runs at a time. Within
    the task, items are processed in parallel using a Semaphore
    (controlled by max_workers or ``settings.task_max_workers``,
    default 3). After the first pass, failed items are automatically
    retried up to ``settings.task_max_retries`` times (default 1).

    Supports cooperative cancellation via ``stop_task``: a cancellation
    flag is checked before each item and between retry rounds.  If the
    flag is set, remaining items are skipped and the task is marked
    'cancelled'.

    Args:
        task_id: The MCP ``tasks.id`` to execute.
        max_workers: Override for parallelism (None → use settings).
        manager: Shared ``RuntimeManager`` carrying the per-item
            ``RuntimeRun`` objects created in ``routes/task.py:create_task``.
            If None (e.g. resume path), a fresh one is created and the
            mirror runs cannot replay events.
        batch_run_ids: List of ``{"seq", "run_id", "operator_name"}`` dicts
            mapping MCP task_items.seq → RuntimeManager run_id. Required
            for event mirroring. If None, mirror writes are skipped.
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
            if manager is None:
                manager = RuntimeManager()

            # Set task to running
            await mcp.update_task_status(task_id, "running")

            # Get all pending items
            items = await mcp.get_pending_task_items(task_id)

            # Build {seq → run_id} lookup so _process_item can find the
            # matching RuntimeRun. Missing entries (e.g. resume path that
            # never went through the new create_task flow) get None and
            # fall back to _process_item_no_mirror.
            run_id_by_seq: dict[int, str] = {}
            if batch_run_ids:
                run_id_by_seq = {b["seq"]: b["run_id"] for b in batch_run_ids}

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
                    seq = item.get("seq")
                    rid = run_id_by_seq.get(seq) if seq is not None else None
                    if rid is None:
                        # No mirror entry: run the pipeline without a registered
                        # RuntimeRun so we still update MCP state, but skip
                        # pipeline_runs / pipeline_events writes.
                        await _process_item_no_mirror(item, graph, mcp, task_id)
                    else:
                        await _process_item(item, graph, mcp, task_id, rid, manager)
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
    finally:
        # Always clean up the registries, even if cancelled via CancelledError
        _cancel_flags.pop(task_id, None)
        _task_handles.pop(task_id, None)


async def retry_failed_task(task_id: int) -> dict:
    """Retry failed operators from a completed task.

    Extracts all failed items from the original task, creates a new
    task containing only those items, registers a fresh RuntimeRun for
    each in a new RuntimeManager (so the retry shows up in the frontend
    task list independently of the original), and kicks off background
    execution.

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

    # Mirror the retry task into pipeline_runs + register RuntimeRuns so
    # the retry shows up in the frontend as independent tasks (the
    # original failed batch's mirror rows are left in place as history).
    manager: RuntimeManager | None = None
    batch_run_ids: list[dict] = []
    try:
        manager = RuntimeManager()
        task_name_prefix = f"[batch {new_task_id}] "
        for item in items:
            op = item["operator_name"]
            run = manager.create_run(op)
            db_create_run(
                run.run_id,
                op,
                hashlib.sha256(op.encode()).hexdigest(),
                task_type="constraint_extract",
                task_name=f"{task_name_prefix}{op}",
            )
            batch_run_ids.append({
                "seq": item["seq"],
                "run_id": run.run_id,
                "operator_name": op,
            })
    except Exception:
        logger.exception(
            "Failed to mirror retry task %s into pipeline_runs", new_task_id
        )
        manager = None
        batch_run_ids = []

    # Kick off background execution with concurrency=1 for retry safety
    asyncio.create_task(
        run_task(new_task_id, max_workers=1,
                manager=manager, batch_run_ids=batch_run_ids)
    )

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
    The resume path does not have a batch_run_ids mapping (the original
    RuntimeManager state was lost across restarts), so per-item runs go
    through the no-mirror fallback in run_task.

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
