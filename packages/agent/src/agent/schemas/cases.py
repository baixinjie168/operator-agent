"""Pydantic schemas for the ``/api/v1/cases`` and ``/api/v1/generator`` endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateCasesRequest(BaseModel):
    """Request body for ``POST /api/v1/cases/generate`` (sync, one-shot)."""

    operator_name: str = Field(..., min_length=1, description="Operator name (e.g. 'aclnnAdaLayerNorm').")
    count: int = Field(default=10, ge=0, le=10_000, description="Number of test cases to generate.")
    seed: int | None = Field(default=None, description="Random seed for reproducibility.")


class GenerateCasesResponse(BaseModel):
    """Response body for ``POST /api/v1/cases/generate``."""

    success: bool
    operator_name: str
    cases_count: int = 0
    output_path: str | None = None
    error: str | None = None


class GetCasesResponse(BaseModel):
    """Response body for ``GET /api/v1/cases/{operator_name}``."""

    operator_name: str
    found: bool
    cases: list[dict] = Field(default_factory=list)
    output_path: str | None = None


class GeneratorRunRequest(BaseModel):
    """Request body for ``POST /api/v1/generator/run`` (async, 5-step pipeline)."""

    operator_name: str = Field(..., min_length=1, description="Operator name (e.g. 'aclnnAdaLayerNorm').")
    count: int = Field(default=10, ge=0, le=10_000, description="Number of test cases to generate.")
    seed: int | None = Field(default=None, description="Random seed for reproducibility.")


class GeneratorRunResponse(BaseModel):
    """Response body for ``POST /api/v1/generator/run``.

    The pipeline runs asynchronously; subscribe to
    ``GET /api/v1/runs/{task_id}/stream`` for real-time SSE progress.
    """

    success: bool
    task_id: str
    operator_name: str
    count: int
    error: str | None = None


class ExecuteRunRequest(BaseModel):
    """Request body for ``POST /api/v1/execute/run`` (async, 3-step pipeline)."""

    operator_name: str = Field(..., min_length=1, description="Operator name (e.g. 'aclnnAdaLayerNorm').")
    cases_json: str = Field(..., min_length=2, description="Test cases JSON array string.")
    server_id: int | None = Field(default=None, description="Server ID for remote execution. If not provided, uses local execution.")


class ExecuteRunResponse(BaseModel):
    """Response body for ``POST /api/v1/execute/run``.

    The pipeline runs asynchronously; subscribe to
    ``GET /api/v1/runs/{task_id}/stream`` for real-time SSE progress.
    """

    success: bool
    task_id: str
    operator_name: str
    cases_count: int = 0
    error: str | None = None
