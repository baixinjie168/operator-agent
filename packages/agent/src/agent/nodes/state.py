"""Pipeline state definition for the deterministic document processing graph."""

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """State flowing through the document processing pipeline."""

    operator_id: int
    doc_id: int
    operator_name: str
    version: int
    content: str
    content_hash: str
    status: str  # "new" | "unchanged" | "updated"
    sections: list[dict[str, Any]]
    parameters: list[dict[str, Any]]
    product_support: list[dict[str, Any]]
    cann_version: str | None
    error: str | None
