"""Task routes: list docs, create tasks, query tasks, download results."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, Query
from fastapi.responses import Response

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.schemas.task import (
    CreateTaskRequest,
    CreateTaskResponse,
    RetryTaskResponse,
    TaskDetailResponse,
    TaskDocItem,
    TaskDocsResponse,
    TaskItemDetail,
    TaskListResponse,
    TaskSummary,
)
from agent.services.task_engine import resume_task, retry_failed_task, retry_item, run_task, stop_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["task"])

_mcp_client = MCPClient()

# Cached parallel config loaded from task_config.yaml
_parallel_config_cache: dict | None = None


def _get_max_workers() -> int:
    """Read max_workers from task_config.yaml (cached).

    Falls back to settings.task_max_workers if the config file is
    missing or doesn't specify parallel.max_workers.
    """
    global _parallel_config_cache
    if _parallel_config_cache is None:
        config_path = Path(settings.task_config_file)
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    _parallel_config_cache = yaml.safe_load(f) or {}
            except Exception:
                logger.warning("Failed to load task_config.yaml, using default max_workers")
                _parallel_config_cache = {}
        else:
            _parallel_config_cache = {}

    parallel_cfg = _parallel_config_cache.get("parallel", {})
    return parallel_cfg.get("max_workers", settings.task_max_workers)


@router.get("/task-docs", response_model=TaskDocsResponse)
async def list_task_docs(search: str | None = Query(default=None)) -> TaskDocsResponse:
    """Scan operators/ directory and return all .md files."""
    ops_dir = Path(settings.operators_dir)
    if not ops_dir.exists():
        return TaskDocsResponse(documents=[], total=0)

    docs: list[TaskDocItem] = []
    for md_file in sorted(ops_dir.rglob("*.md")):
        name = md_file.stem
        if search and search.lower() not in name.lower():
            continue
        rel_path = str(md_file.relative_to(ops_dir.parent))
        # category = parent dir relative to operators/, or "" if in root
        parts = md_file.relative_to(ops_dir).parts
        category = parts[0] if len(parts) > 1 else ""
        docs.append(
            TaskDocItem(
                name=name,
                path=rel_path,
                size=md_file.stat().st_size,
                category=category,
            )
        )

    return TaskDocsResponse(documents=docs, total=len(docs))


@router.get("/task-categories")
async def list_task_categories() -> dict:
    """List operator sub-directories (categories) with their .md document counts.

    This is a lightweight endpoint for the folder-selection panel in the
    UI prototype — it returns only category names and counts rather than
    the full document list.
    """
    ops_dir = Path(settings.operators_dir)
    if not ops_dir.exists():
        return {"categories": [], "total": 0}

    categories: list[dict] = []
    total = 0

    # Sub-directories
    for sub in sorted(ops_dir.iterdir(), key=lambda p: p.name):
        if sub.is_dir():
            count = sum(1 for _ in sub.rglob("*.md"))
            if count > 0:
                categories.append({"name": sub.name, "count": count})
                total += count

    # Loose .md files in the operators root (category = "")
    root_count = sum(1 for _ in ops_dir.glob("*.md"))
    if root_count > 0:
        categories.append({"name": "", "count": root_count})
        total += root_count

    return {"categories": categories, "total": total}


@router.post("/tasks", response_model=CreateTaskResponse)
async def create_task(req: CreateTaskRequest) -> CreateTaskResponse:
    """Create a new batch task, copy files, and start background execution.

    Accepts either ``file_paths`` (individual file paths relative to the
    project root) or ``categories`` (operator sub-directory names under
    ``operators/``).  When ``categories`` is provided, all ``.md`` files
    under each sub-directory are collected automatically.  Both fields may
    be combined; the resulting file list is de-duplicated.
    """
    ops_base = Path(settings.operators_dir).parent  # project root
    ops_dir = Path(settings.operators_dir)

    # Resolve file paths from categories + explicit file_paths
    file_paths: list[str] = []
    seen: set[str] = set()

    if req.categories:
        for cat in req.categories:
            cat_dir = ops_dir / cat
            if cat_dir.is_dir():
                for md_file in sorted(cat_dir.rglob("*.md")):
                    rel = str(md_file.relative_to(ops_base))
                    if rel not in seen:
                        file_paths.append(rel)
                        seen.add(rel)
            else:
                logger.warning("Category directory not found: %s", cat_dir)

    for fp in req.file_paths:
        if fp not in seen:
            file_paths.append(fp)
            seen.add(fp)

    if not file_paths:
        return CreateTaskResponse(
            success=False, error="No files selected (file_paths and categories are both empty)"
        )

    # Generate timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir_name = f"uploads/{timestamp}"
    upload_dir = Path(upload_dir_name)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate task name
    task_name = req.name or f"batch-{timestamp}"

    # Copy files
    for fp in file_paths:
        src = ops_base / fp
        if not src.exists():
            logger.warning("File not found: %s", src)
            continue
        dst = upload_dir / src.name
        # Use plain read/write instead of shutil.copy to avoid
        # PermissionError on WSL/NTFS (chmod/utime not permitted).
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Extract operator names from filenames
    items = []
    for seq, fp in enumerate(file_paths, start=1):
        src = ops_base / fp
        if not src.exists():
            continue
        operator_name = _extract_operator_name(src)
        items.append(
            {
                "seq": seq,
                "operator_name": operator_name,
                "file_path": str(upload_dir / src.name),
            }
        )

    if not items:
        return CreateTaskResponse(success=False, error="No valid files found")

    try:
        # Create task in DB
        result = await _mcp_client.create_task(task_name, len(items), upload_dir_name)
        task_id = result["task_id"]

        # Create task items
        await _mcp_client.create_task_items(task_id, items)

        # Start background execution
        asyncio.create_task(run_task(task_id, max_workers=_get_max_workers()))

        return CreateTaskResponse(
            success=True,
            task_id=task_id,
            name=task_name,
            total_count=len(items),
            upload_dir=upload_dir_name,
            status="pending",
        )

    except Exception as e:
        logger.exception("Failed to create task")
        return CreateTaskResponse(success=False, error=str(e))


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks() -> TaskListResponse:
    """List all tasks."""
    try:
        result = await _mcp_client.list_tasks()
        tasks = [
            TaskSummary(
                id=t["id"],
                name=t["name"],
                status=t["status"],
                total_count=t["total_count"],
                completed_count=t["completed_count"],
                failed_count=t["failed_count"],
                created_at=t.get("created_at"),
                updated_at=t.get("updated_at"),
            )
            for t in result
        ]
        return TaskListResponse(tasks=tasks)
    except Exception:
        return TaskListResponse(tasks=[])


@router.get("/tasks-summary")
async def get_tasks_summary() -> dict:
    """Aggregated statistics across all tasks.

    Returns two levels of aggregation:

    * ``totals`` — *item-level* counts (sum of completed/failed operator
      documents across every task) and overall success rate.
    * ``task_counts`` — *task-level* status counts (how many tasks are
      pending / running / completed / failed / cancelled).  This drives
      the summary cards in the UI prototype.
    """
    try:
        tasks = await _mcp_client.list_tasks()
    except Exception:
        tasks = []

    rows: list[dict] = []
    total_all = 0
    completed_all = 0
    failed_all = 0
    # Task-level status counts
    task_counts = {
        "total": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "pending": 0,
        "cancelled": 0,
    }
    for t in tasks:
        total = t.get("total_count", 0)
        completed = t.get("completed_count", 0)
        failed = t.get("failed_count", 0)
        rate = round(completed / total * 100, 1) if total > 0 else 0.0
        rows.append({
            "id": t["id"],
            "name": t["name"],
            "status": t["status"],
            "total_count": total,
            "completed_count": completed,
            "failed_count": failed,
            "success_rate": rate,
            "created_at": t.get("created_at"),
        })
        total_all += total
        completed_all += completed
        failed_all += failed

        status = t.get("status", "pending")
        task_counts["total"] += 1
        if status in task_counts:
            task_counts[status] += 1

    overall_rate = round(completed_all / total_all * 100, 1) if total_all > 0 else 0.0
    return {
        "tasks": rows,
        "totals": {
            "task_count": len(rows),
            "total_count": total_all,
            "completed_count": completed_all,
            "failed_count": failed_all,
            "success_rate": overall_rate,
        },
        "task_counts": task_counts,
    }


@router.get("/tasks/{task_id}", response_model=TaskDetailResponse)
async def get_task_detail(task_id: int) -> TaskDetailResponse:
    """Get task detail with all items."""
    result = await _mcp_client.get_task_with_items(task_id)
    if result is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    items = [
        TaskItemDetail(
            id=item["id"],
            seq=item["seq"],
            operator_name=item["operator_name"],
            file_path=item["file_path"],
            status=item["status"],
            doc_id=item.get("doc_id"),
            error=item.get("error"),
            started_at=item.get("started_at"),
            finished_at=item.get("finished_at"),
        )
        for item in result.get("items", [])
    ]

    return TaskDetailResponse(
        id=result["id"],
        name=result["name"],
        status=result["status"],
        total_count=result["total_count"],
        completed_count=result["completed_count"],
        failed_count=result["failed_count"],
        upload_dir=result["upload_dir"],
        created_at=result.get("created_at"),
        updated_at=result.get("updated_at"),
        items=items,
    )


from agent.utils.file_utils import extract_operator_name_from_file as _extract_operator_name


@router.get("/tasks/{task_id}/download")
async def download_task_results(task_id: int) -> Response:
    """Download all completed operator results for a task as a ZIP file.

    Each completed item's json_constraints is written as {operator_name}.json
    inside the archive.
    """
    result = await _mcp_client.get_task_with_items(task_id)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")

    task_name = result.get("name", f"task-{task_id}")
    items = result.get("items", [])

    buf = io.BytesIO()
    included = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            if item.get("status") != "completed":
                continue
            operator_name = item.get("operator_name", "")
            if not operator_name:
                continue
            try:
                jc = await _mcp_client.get_json_constraints(operator_name)
            except Exception:
                logger.warning("Failed to fetch json_constraints for %s", operator_name)
                continue
            if not jc:
                continue
            content = json.dumps(jc, ensure_ascii=False, indent=2)
            zf.writestr(f"{operator_name}.json", content)
            included += 1

    if included == 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No completed results to download")

    buf.seek(0)
    filename = f"{task_name}-results.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/tasks/{task_id}/resume")
async def resume_stuck_task(task_id: int) -> dict:
    """Resume a task that has items stuck in 'running' status.

    Resets stuck items to 'pending' and re-starts background execution.
    """
    # Verify task exists
    task = await _mcp_client.get_task(task_id)
    if task is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    try:
        result = await resume_task(task_id)
        return {
            "success": True,
            "task_id": task_id,
            "reset_count": result["reset_count"],
        }
    except Exception as e:
        logger.exception("Failed to resume task %s", task_id)
        return {"success": False, "error": str(e)}


@router.post("/tasks/{task_id}/stop")
async def stop_running_task(task_id: int) -> dict:
    """Stop a running task.

    Cancels background execution, resets in-progress items to 'pending',
    and marks the task as 'cancelled'.  Only tasks with status 'running'
    can be stopped.
    """
    # Verify task exists
    task = await _mcp_client.get_task(task_id)
    if task is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    try:
        result = await stop_task(task_id)
        return {
            "success": True,
            "task_id": task_id,
            "reset_count": result["reset_count"],
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("Failed to stop task %s", task_id)
        return {"success": False, "error": str(e)}


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: int) -> dict:
    """Delete a task and all associated operator data."""
    task = await _mcp_client.get_task(task_id)
    if task is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        result = await _mcp_client.delete_task(task_id)
        return {"success": True, **result}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("Failed to delete task %s", task_id)
        return {"success": False, "error": str(e)}


@router.post("/tasks/{task_id}/retry-failed", response_model=RetryTaskResponse)
async def retry_failed_operators(task_id: int) -> RetryTaskResponse:
    """Retry all failed operators from a completed task.

    Creates a new task containing only the operators that failed in the
    original task, then kicks off background execution for the new task.
    """
    # Verify task exists
    task = await _mcp_client.get_task(task_id)
    if task is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    try:
        result = await retry_failed_task(task_id)
        return RetryTaskResponse(
            success=True,
            new_task_id=result["new_task_id"],
            new_task_name=result["new_task_name"],
            failed_count=result["failed_count"],
            upload_dir=result["upload_dir"],
        )
    except ValueError as e:
        return RetryTaskResponse(success=False, error=str(e))
    except Exception as e:
        logger.exception("Failed to retry task %s", task_id)
        return RetryTaskResponse(success=False, error=str(e))


@router.post("/tasks/{task_id}/items/{item_id}/retry")
async def retry_single_item(task_id: int, item_id: int) -> dict:
    """Retry a single failed task item.

    Resets the item to 'pending' and re-runs the task.  Only allowed when
    the task is not currently running.
    """
    task = await _mcp_client.get_task(task_id)
    if task is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    try:
        result = await retry_item(task_id, item_id)
        return {"success": True, **result}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.exception("Failed to retry item %s in task %s", item_id, task_id)
        return {"success": False, "error": str(e)}


@router.post("/tasks/{task_id}/check-constraints")
async def check_task_constraints(task_id: int) -> dict:
    """Run constraint check for all completed operators in a task.

    Iterates through all task items with status='completed' and doc_id,
    runs the constraint check agent for each, and saves HTML reports.
    """
    from agent.nodes.constraint_check_agent import run_constraint_check

    task = await _mcp_client.get_task(task_id)
    if task is None:
        return {"success": False, "error": "Task not found"}

    items = await _mcp_client.get_task_items(task_id)
    completed_items = [
        it for it in items
        if it.get("status") == "completed" and it.get("doc_id")
    ]

    if not completed_items:
        logger.info("Task %s: no completed operators to check", task_id)
        return {"success": False, "error": "No completed operators to check"}

    logger.info(
        "Task %s: starting constraint check for %d operators: %s",
        task_id, len(completed_items),
        ", ".join(it.get("operator_name", "?") for it in completed_items),
    )

    checked = 0
    errors: list[dict] = []
    for i, item in enumerate(completed_items):
        op_name = item.get("operator_name", "?")
        logger.info("Task %s: checking %s (%d/%d)...", task_id, op_name, i + 1, len(completed_items))
        try:
            doc = await _mcp_client.get_doc_for_check(item["doc_id"])
            if not doc:
                logger.warning("Task %s: no doc data for %s, skipping", task_id, op_name)
                continue
            json_constraints = doc.get("json_constraints", "{}")
            content = doc.get("content", "")
            operator_name = doc.get("operator_name") or op_name
            if not content.strip() or json_constraints == "{}":
                logger.warning("Task %s: empty content/constraints for %s, skipping", task_id, op_name)
                continue

            html = await run_constraint_check(
                content, json_constraints, operator_name,
            )
            if html:
                await _mcp_client.save_constraint_check_report(item["doc_id"], html)
                checked += 1
                logger.info("Task %s: %s done (%d/%d)", task_id, op_name, checked, len(completed_items))
            else:
                errors.append({"name": op_name, "error": "Check produced no output"})
                logger.warning("Task %s: %s produced no output", task_id, op_name)
        except Exception as e:
            errors.append({"name": op_name, "error": str(e)})
            logger.warning("Task %s: %s failed: %s", task_id, op_name, e, exc_info=True)

    logger.info(
        "Task %s: constraint check complete: %d/%d checked, %d errors",
        task_id, checked, len(completed_items), len(errors),
    )

    return {
        "success": len(errors) == 0,
        "checked": checked,
        "total": len(completed_items),
        "errors": errors,
    }
