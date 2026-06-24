"""Task routes: list docs, create tasks, query tasks, download results."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

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
from agent.services.task_engine import resume_task, retry_failed_task, run_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["task"])

_mcp_client = MCPClient()


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


@router.post("/tasks", response_model=CreateTaskResponse)
async def create_task(req: CreateTaskRequest) -> CreateTaskResponse:
    """Create a new batch task, copy files, and start background execution."""
    if not req.file_paths:
        return CreateTaskResponse(success=False, error="file_paths is empty")

    # Generate timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir_name = f"uploads/{timestamp}"
    upload_dir = Path(upload_dir_name)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate task name
    task_name = req.name or f"batch-{timestamp}"

    # Copy files
    ops_base = Path(settings.operators_dir).parent  # project root
    for fp in req.file_paths:
        src = ops_base / fp
        if not src.exists():
            logger.warning("File not found: %s", src)
            continue
        dst = upload_dir / src.name
        shutil.copy2(str(src), str(dst))

    # Extract operator names from filenames
    items = []
    for seq, fp in enumerate(req.file_paths, start=1):
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
        asyncio.create_task(run_task(task_id))

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
    """Aggregated success/failure statistics across all tasks."""
    try:
        tasks = await _mcp_client.list_tasks()
    except Exception:
        tasks = []

    rows: list[dict] = []
    total_all = 0
    completed_all = 0
    failed_all = 0
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


def _extract_operator_name(file_path: Path) -> str:
    """Extract operator name from filename (stem).

    E.g. aclnnAddRmsNorm.md -> aclnnAddRmsNorm
    """
    stem = file_path.stem
    # Try to read the file and extract from H1 heading
    try:
        content = file_path.read_text(encoding="utf-8")
        for line in content.split("\n"):
            m = re.match(r"^#{1,2}\s+(.+?)-CANN社区版", line)
            if m:
                return m.group(1).strip()
            m = re.match(r"^#{1,2}\s+(aclnn?\w+)", line)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return stem


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
