"""Sub-graph state for BuildParamConstraint."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from agent.nodes.state import merge_errors


class BuildParamConstraintState(TypedDict, total=False):
    """State flowing through the build_param_constraint sub-graph.

    Fields shared with PipelineState (doc_id, operator_name, error) are
    automatically mapped by LangGraph when the sub-graph is registered as a
    node in the parent graph.

    String keys use "::" as separator to survive JSON round-trips:
    - dimensions_map:    "fn::pn::shape_text" -> list
    - allowed_range_map: "fn::pn" -> {"type": "range"|"enum", "value": [[min,max], ...]}
    - attrs_map:         "fn::pn::plat" -> dict
    """

    # -- from PipelineState --
    doc_id: int
    operator_name: str

    # -- fetch_param_data output --
    params: list[dict[str, Any]]
    sig_type_map: dict[str, str]  # "fn::pn" -> C type
    all_sig_param_names: list[str]
    dtype_by_platform: dict[str, dict[str, list[str]]]  # lists (not sets)
    supported_platforms: list[str]
    constraints_text: str
    param_relations: list[dict[str, Any]]

    # -- dimensions_build output --
    dimensions_map: dict[str, list]  # "fn::pn::shape_text" -> dimensions array

    # -- allowed_range_build output --
    allowed_range_map: dict[str, dict]  # "fn::pn" -> {"type": "range"|"enum", "value": [[min,max], ...]}

    # -- attrs_build output --
    attrs_map: dict[str, dict[str, Any]]  # "fn::pn::plat" -> {dtype, format, ...}

    # -- constraint_assemble output --
    constraint_updates: list[dict[str, Any]]

    # error (same reducer as parent)
    error: Annotated[str | None, merge_errors]
