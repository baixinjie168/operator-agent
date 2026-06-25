"""Pipeline state definition for the deterministic document processing graph."""

from typing import Annotated, Any, TypedDict


def merge_errors(current: str | None, new: str | None) -> str | None:
    """Reducer for parallel node error merging: concatenate non-None errors."""
    if current is None:
        return new
    if new is None:
        return current
    return f"{current}; {new}"


def last_value(current: Any, new: Any) -> Any:
    """Reducer: last write wins."""
    return new


class PipelineState(TypedDict, total=False):
    """State flowing through the document processing pipeline."""

    run_id: str  # Current task/run ID for database operations
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
    single_param_constraints: list[dict[str, Any]]
    implicit_params: list[dict[str, Any]]
    platform_constants: list[dict[str, Any]]
    # ── ExtractorAgent frontend-facing outputs ──
    # build_param_relations emits these for the ExtractorAgent constraint
    # detail panel (cd-rel-check / cd-cst-check 三段式校验视图).
    validation_results: list[dict[str, Any]]  # build_param_relations output
    cst_validation_results: list[dict[str, Any]]  # build_param_constraint output
    relations_count: int
    platforms_count: int
    params_count: int
    dimensions_count: int
    range_count: int
    error: Annotated[str | None, merge_errors]
    # ── GeneratorAgent output (set by case_subgraph nodes) ──
    # Loaded by case_match_model
    constraints_raw: dict[str, Any] | None
    # Counters from case_init_static
    sampled_shapes: int
    sampled_dtypes: int
    # Counter from case_solve_constraints
    valid_combos: int
    rejected_combos: int
    # Final outputs from case_generate
    cases: list[dict[str, Any]]
    cases_path: str | None
    cases_count: int | None
    cases_seed: int | None
    # ── ExecuterAgent output (set by executer_subgraph nodes) ──
    atk_executor_path: Annotated[str | None, last_value]
    atk_executor_code: Annotated[str, last_value]
    exec_result: dict[str, Any]
    # ── ATK execution parameters (forwarded to remote ATK command) ──
    task_type: str  # e.g. "precision" / "performance" / "function"
    execution_count: int  # -e flag value passed to ``atk task``
    # ── Server info for remote execution ──
    server_info: dict[str, Any] | None
