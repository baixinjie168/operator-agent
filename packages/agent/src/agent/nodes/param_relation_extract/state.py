"""Sub-graph state for parameter relation extraction."""

from typing import Annotated, Any, TypedDict

from agent.nodes.state import merge_errors


class RelationExtractState(TypedDict, total=False):
    doc_id: int
    operator_name: str
    ws_section_content: str
    exe_section_content: str
    ws_relations: list[dict[str, Any]]
    exe_relations: list[dict[str, Any]]
    merged_relations: list[dict[str, Any]]
    error: Annotated[str | None, merge_errors]
