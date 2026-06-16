"""Sub-graph state for parameter relation extraction."""

from typing import Annotated, Any, TypedDict

from agent.nodes.state import merge_errors


def merge_coverage_reports(
    current: dict[str, Any] | None, new: dict[str, Any] | None
) -> dict[str, Any]:
    """Reducer for parallel coverage_report updates from ws/exe nodes.

    Each node returns {"ws": report} or {"exe": report}.
    This reducer merges them into {"ws": ws_report, "exe": exe_report}.
    """
    if current is None:
        return new or {}
    if new is None:
        return current
    # Merge: combine ws and exe keys
    return {**current, **new}


class RelationExtractState(TypedDict, total=False):
    doc_id: int
    operator_name: str
    ws_section_content: str
    exe_section_content: str
    param_names: list[str]
    implicit_params: list[dict[str, Any]]
    ws_relations: list[dict[str, Any]]
    exe_relations: list[dict[str, Any]]
    merged_relations: list[dict[str, Any]]
    coverage_report: Annotated[dict[str, Any] | None, merge_coverage_reports]
    error: Annotated[str | None, merge_errors]
