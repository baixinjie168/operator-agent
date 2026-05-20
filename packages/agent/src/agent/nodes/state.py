"""Pipeline state definition for the deterministic document processing graph."""

from __future__ import annotations

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """State flowing through the InitDoc → ParseParams → PersistParams pipeline."""

    operator_id: int
    operator_name: str
    version: int
    content: str
    sections: list[dict[str, Any]]
    parameters: list[dict[str, Any]]
    error: str | None
