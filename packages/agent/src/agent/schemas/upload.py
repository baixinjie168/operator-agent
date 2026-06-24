from pydantic import BaseModel


class UploadResponse(BaseModel):
    success: bool
    operator_name: str | None = None
    cann_version: str | None = None
    status: str | None = None  # "new" | "unchanged" | "updated"
    version: int | None = None
    sections_count: int | None = None
    task_id: str | None = None
    error: str | None = None
