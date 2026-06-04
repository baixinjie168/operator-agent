"""Pipeline state definition for the deterministic document processing graph."""

from typing import Annotated, Any, TypedDict


def merge_errors(current: str | None, new: str | None) -> str | None:
    """Reducer for parallel node error merging: concatenate non-None errors."""
    if current is None:
        return new
    if new is None:
        return current
    return f"{current}; {new}"


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
    function_explanation_summary: dict[str, Any]
    error: Annotated[str | None, merge_errors]
