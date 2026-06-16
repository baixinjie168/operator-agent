"""Pydantic schemas for the batch task feature."""

from pydantic import BaseModel


class TaskDocItem(BaseModel):
    """A single operator document available for task creation."""

    name: str
    path: str
    size: int
    category: str = ""


class TaskDocsResponse(BaseModel):
    """Response for GET /api/v1/task-docs."""

    documents: list[TaskDocItem]
    total: int


class CreateTaskRequest(BaseModel):
    """Request body for POST /api/v1/tasks."""

    name: str | None = None
    file_paths: list[str]


class CreateTaskResponse(BaseModel):
    """Response for POST /api/v1/tasks."""

    success: bool
    task_id: int | None = None
    name: str | None = None
    total_count: int | None = None
    upload_dir: str | None = None
    status: str | None = None
    error: str | None = None


class TaskSummary(BaseModel):
    """Summary of a task for list view."""

    id: int
    name: str
    status: str
    total_count: int
    completed_count: int
    failed_count: int
    created_at: str | None = None
    updated_at: str | None = None


class TaskListResponse(BaseModel):
    """Response for GET /api/v1/tasks."""

    tasks: list[TaskSummary]


class RetryTaskResponse(BaseModel):
    """Response for POST /api/v1/tasks/{task_id}/retry-failed."""

    success: bool
    new_task_id: int | None = None
    new_task_name: str | None = None
    failed_count: int | None = None
    upload_dir: str | None = None
    error: str | None = None


class TaskItemDetail(BaseModel):
    """Detail of a single task item."""

    id: int
    seq: int
    operator_name: str
    file_path: str
    status: str
    doc_id: int | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class TaskDetailResponse(BaseModel):
    """Response for GET /api/v1/tasks/{task_id}."""

    id: int
    name: str
    status: str
    total_count: int
    completed_count: int
    failed_count: int
    upload_dir: str
    created_at: str | None = None
    updated_at: str | None = None
    items: list[TaskItemDetail]
