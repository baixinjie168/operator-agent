"""Sub-graph state for llm_description extraction."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from agent.nodes.state import merge_errors


class DescriptionExtractState(TypedDict, total=False):
    """State flowing through the llm_description extraction sub-graph.

    Fields shared with :class:`PipelineState` (``doc_id``, ``operator_name``,
    ``parameters``, ``error``) are automatically mapped by LangGraph when the
    sub-graph is registered as a node in the parent graph.

    ``parameters`` is both an input (from the parent) and an output (written
    back as the enriched list after extraction).
    """

    # ── from PipelineState ────────────────────────────────────────────────
    doc_id: int
    operator_name: str
    parameters: list[dict[str, Any]]

    # ── fetch_sections output ─────────────────────────────────────────────
    ws_sections_text: str
    exe_sections_text: str

    # ── extract_ws / extract_exe output ───────────────────────────────────
    ws_results: list[dict[str, Any]]
    exe_results: list[dict[str, Any]]

    # ── validate_results output ───────────────────────────────────────────
    validation_report: dict[str, Any]
    coverage_report: dict[str, Any]

    # verify_and_enhance output
    enhance_count: int

    # error (same reducer as parent)──────────────────────────────────
    error: Annotated[str | None, merge_errors]

