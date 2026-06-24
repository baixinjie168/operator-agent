"""LangGraph pipeline graph -- unified build_pipeline() entry point.

The pipeline is composed of three independent stages that can be freely
combined:

    EXTRACT  : DocProcessorAgent + ExtractorAgent (constraint extraction)
    GENERATE : GeneratorAgent (test case generation)
    EXECUTE  : ExecuterAgent (test case execution)

Usage::

    build_pipeline(["extract"])                        # extract only
    build_pipeline(["extract", "generate"])            # extract + generate
    build_pipeline(["generate"])                       # generate only
    build_pipeline(["generate", "execute"])            # generate + execute
    build_pipeline(["extract", "generate", "execute"]) # full pipeline

Prerequisites are checked automatically:
  - GENERATE without EXTRACT: case_match_model loads constraints from DB;
    returns error if constraints do not exist.
  - EXECUTE without GENERATE: exec_generate_atk requires cases_path in
    state; returns error if not set.

EXTRACT stage flow (main branch architecture):
    InitDoc -> [ProductSupport || FunctionSignatureExtract || FunctionExplanationExtract]
           -> TableColumnExtract
           -> LlmDescriptionExtract (subgraph)
           -> [detail extractors in parallel]
                (param_relation_extract is a subgraph: implicit_param_extract
                 + extract_ws + extract_exe + parameter_representation_build
                 + merge_relations + save_relations)
           -> BuildParamRelations -> BuildSingleParamConstraint
           -> BuildParamConstraint (subgraph) -> AssembleResult -> END

Note: FunctionSignatureExtract also produces the flat parameters list
(replaces the old parse_params node).
"""

from __future__ import annotations

import logging
from enum import Enum

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# -- Extract stage nodes (using main branch new architecture) --
from agent.nodes.allowed_range_extract import allowed_range_extract_node as _allowed_range_extract
from agent.nodes.array_length_extract import array_length_extract_node as _array_length_extract
from agent.nodes.assemble_result import assemble_result_node as _assemble_result
from agent.nodes.build_param_constraint import (
    create_build_param_constraint_subgraph,
)
from agent.nodes.build_param_relations import build_param_relations_node as _build_param_relations
from agent.nodes.determinism_extract import determinism_extract_node as _determinism_extract
from agent.nodes.dformat_extract import dformat_extract_node as _dformat_extract
from agent.nodes.dtype_combo_extract import dtype_combo_extract_node as _dtype_combo_extract
from agent.nodes.dtype_extract import dtype_extract_node as _dtype_extract
from agent.nodes.function_explanation_extract import (
    function_explanation_extract_node as _function_explanation_extract,
)
from agent.nodes.function_signature_extract import function_signature_extract_node as _function_signature_extract
from agent.nodes.init_doc import init_doc_node as _init_doc
from agent.nodes.llm_description_extract import create_description_extract_subgraph
from agent.nodes.optional_extract import optional_extract_node as _optional_extract
from agent.nodes.param_relation_extract import create_param_relation_subgraph
from agent.nodes.product_support import product_support_node as _product_support
from agent.nodes.return_code_extract import return_code_extract_node as _return_code_extract
from agent.nodes.shape_extract import shape_extract_node as _shape_extract
from agent.nodes.single_param_constraint import (
    build_single_param_constraint_node as _build_single_param_constraint,
)
from agent.nodes.state import PipelineState
from agent.nodes.table_column_extract import table_column_extract_node as _table_column_extract

# -- Generate stage nodes --
from agent.nodes.case_subgraph import (
    case_generate_node as _case_generate,
    case_init_static_node as _case_init_static,
    case_match_model_node as _case_match_model,
    case_solve_constraints_node as _case_solve_constraints,
)

# -- Execute stage nodes --
from agent.nodes.executer_subgraph import (
    exec_cpu_derivation_node as _exec_cpu_derivation,
    exec_generate_atk_node as _exec_generate_atk,
    exec_run_atk_node as _exec_run_atk,
)

# -- Runtime tracing --
from agent.runtime import traced_node
from agent.runtime.decorators import _AGENT_MAP, _node_done_msg, _node_meta, _node_progress_pct
from agent.runtime.context import get_context, set_context
from agent.runtime.events import EventType, SpanStatus, SpanType

logger = logging.getLogger(__name__)


def traced_subgraph(node_id: str):
    """Wrap a compiled LangGraph subgraph with the same SSE/span lifecycle as
    :func:`traced_node`.

    The default ``StateGraph.add_node(name, subgraph)`` pattern invokes the
    subgraph directly, which means **no** ``NODE_START`` / ``NODE_SUCCESS``
    events are emitted and the parent graph state mutations are invisible to
    the runtime observability system. The frontend ExtractorAgent constraint
    detail panel relies on ``node.completed`` events with
    ``d.data.output.validation_results`` to populate the
    ``cd-cst-check`` 三段式校验视图, so the subgraph must be wrapped.

    Usage::

        bpc_subgraph = create_build_param_constraint_subgraph()
        graph.add_node(
            "build_param_constraint",
            traced_subgraph("build_param_constraint")(bpc_subgraph),
        )
    """
    import asyncio

    def decorator(subgraph):
        # NOTE: do NOT use @functools.wraps here — CompiledStateGraph exposes
        # ``__call__`` as a non-callable descriptor (since it's an instance
        # of a class with a metaclass), and ``functools.wraps`` ends up
        # invoking ``type.__call__`` on the subgraph, which raises
        # ``TypeError: descriptor '__call__' for 'type' objects doesn't
        # apply to a 'CompiledStateGraph' object``. The wrapper below is a
        # plain async function, which is what LangGraph expects for a node.
        async def wrapper(state, config=None):
            ctx = get_context()
            if ctx is None:
                # No runtime context — run unwrapped, preserve original
                # behaviour for back-compat (e.g. unit tests).
                return await subgraph.ainvoke(state, config=config)

            run = ctx.manager.get_run(ctx.run_id)
            if not run:
                return await subgraph.ainvoke(state, config=config)

            agent_id = _AGENT_MAP.get(node_id, "doc")
            span = ctx.manager.open_span(
                run_id=ctx.run_id,
                parent_span_id=ctx.current_span_id,
                span_type=SpanType.NODE,
                name=node_id,
            )
            ctx.manager.emit(EventType.NODE_START, ctx.run_id, span, {
                "agent_id": agent_id,
                "node_id": node_id,
                "message": f"{node_id} 开始...",
                "step_index": 0,
                "progress_pct": 0,
                "progress_text": "开始",
            })

            try:
                # Subgraph runs with the current node context so internal
                # progress events share the same span hierarchy.
                node_ctx = ctx.__class__(ctx.run_id, ctx.manager)
                node_ctx.trace_id = ctx.trace_id
                node_ctx.current_span_id = span.span_id
                node_ctx.current_node_id = node_id
                set_context(node_ctx)

                result = await subgraph.ainvoke(state, config=config)

                node_status = result.get("status", "") if isinstance(result, dict) else ""
                node_error = result.get("error") if isinstance(result, dict) else None

                if node_status == "unchanged":
                    ctx.manager.close_span(ctx.run_id, span, SpanStatus.SUCCESS, output=result)
                    ctx.manager.emit(EventType.NODE_SKIPPED, ctx.run_id, span, {
                        "agent_id": agent_id,
                        "node_id": node_id,
                        "message": "文档未变更，跳过解析",
                        "progress_pct": 100,
                        "progress_text": "跳过",
                    })
                elif node_status == "error" or (node_error and isinstance(node_error, str)):
                    err_msg = node_error or "节点执行失败"
                    ctx.manager.close_span(ctx.run_id, span, SpanStatus.ERROR, output=result, error=err_msg)
                    ctx.manager.emit(EventType.NODE_ERROR, ctx.run_id, span, {
                        "agent_id": agent_id,
                        "node_id": node_id,
                        "message": err_msg,
                        "error": err_msg,
                    })
                else:
                    ctx.manager.close_span(ctx.run_id, span, SpanStatus.SUCCESS, output=result)
                    ctx.manager.emit(EventType.NODE_SUCCESS, ctx.run_id, span, {
                        "agent_id": agent_id,
                        "node_id": node_id,
                        "message": _node_done_msg(node_id, result),
                        "step_index": 99,
                        "progress_pct": _node_progress_pct(node_id),
                        "progress_text": "完成",
                        "meta": _node_meta(node_id, result),
                        "output": result,
                    })

                # Restore parent context so subsequent sibling nodes see
                # the original span as their parent.
                set_context(ctx)
                await asyncio.sleep(0)
                return result

            except asyncio.CancelledError:
                ctx.manager.close_span(ctx.run_id, span, SpanStatus.ERROR, error="cancelled")
                ctx.manager.emit(EventType.NODE_ERROR, ctx.run_id, span, {
                    "agent_id": agent_id,
                    "node_id": node_id,
                    "message": "节点执行被取消",
                    "error": "cancelled",
                })
                set_context(ctx)
                raise

            except Exception as e:
                logger.exception("Subgraph %s failed", node_id)
                ctx.manager.close_span(ctx.run_id, span, SpanStatus.ERROR, error=str(e))
                ctx.manager.emit(EventType.NODE_ERROR, ctx.run_id, span, {
                    "agent_id": agent_id,
                    "node_id": node_id,
                    "message": str(e),
                    "error": str(e),
                })
                set_context(ctx)
                await asyncio.sleep(0)
                return {"error": str(e)}

        return wrapper

    return decorator


# -------------------------------------------------------------------
#  Stage enum
# -------------------------------------------------------------------

class PipelineStage(str, Enum):
    EXTRACT = "extract"
    GENERATE = "generate"
    EXECUTE = "execute"


# -------------------------------------------------------------------
#  Conditional routers
# -------------------------------------------------------------------

def _should_continue(state: dict) -> list[str]:
    """Route after init_doc: fan out to three nodes, or END on error."""
    status = state.get("status", "new")
    if status in ("error",):
        return END
    return ["product_support", "function_signature_extract", "function_explanation_extract"]


def _route_after_case_match(state: dict) -> str:
    return END if state.get("error") else "case_init_static"


def _route_after_case_init(state: dict) -> str:
    return END if state.get("error") else "case_solve_constraints"


def _route_after_exec_generate(state: dict) -> str:
    return END if state.get("error") else "exec_cpu_derivation"


def _route_after_exec_derivation(state: dict) -> str:
    return END if state.get("error") else "exec_run_atk"


# -------------------------------------------------------------------
#  Stage constants
# -------------------------------------------------------------------

_STAGE_ORDER = [PipelineStage.EXTRACT, PipelineStage.GENERATE, PipelineStage.EXECUTE]

_STAGE_LAST_NODE = {
    PipelineStage.EXTRACT: "assemble_result",
    PipelineStage.GENERATE: "case_generate",
    PipelineStage.EXECUTE: "exec_run_atk",
}

_STAGE_FIRST_NODE = {
    PipelineStage.EXTRACT: "init_doc",
    PipelineStage.GENERATE: "case_match_model",
    PipelineStage.EXECUTE: "exec_generate_atk",
}

_DETAIL_EXTRACTORS = [
    "shape_extract", "dtype_extract", "dformat_extract",
    "optional_extract", "array_length_extract",
    "allowed_range_extract", "return_code_extract", "determinism_extract",
    "dtype_combo_extract", "param_relation_extract",
]


# -------------------------------------------------------------------
#  Stage builders
# -------------------------------------------------------------------

def _build_extract(graph: StateGraph, *, is_first: bool, is_last: bool) -> None:
    """Build the EXTRACT stage using main new architecture.

    The stage flow:
    InitDoc → [ProductSupport ∥ FunctionSignatureExtract
               ∥ FunctionExplanationExtract] → TableColumnExtract
           → LlmDescriptionExtract (subgraph)
           → [ShapeExtract ∥ DtypeExtract ∥ DFormatExtract ∥ OptionalExtract
              ∥ ArrayLengthExtract ∥ AllowedRangeExtract
              ∥ ReturnCodeExtract ∥ DeterminismExtract ∥ DtypeComboExtract
              ∥ ParamRelationExtract (subgraph: implicit_param_extract
                 + extract_ws + extract_exe + parameter_representation_build
                 + merge_relations + save_relations)]
           → BuildParamRelations → BuildSingleParamConstraint
           → BuildParamConstraint (subgraph)
           → AssembleResult → END
    """
    graph.add_node("init_doc", traced_node("init_doc")(_init_doc))
    graph.add_node("product_support", traced_node("product_support")(_product_support))
    graph.add_node("function_signature_extract", traced_node("function_signature_extract")(_function_signature_extract))
    graph.add_node("function_explanation_extract", traced_node("function_explanation_extract")(_function_explanation_extract))
    graph.add_node("table_column_extract", traced_node("table_column_extract")(_table_column_extract))
    graph.add_node("llm_description_extract", traced_subgraph("llm_description_extract")(create_description_extract_subgraph()))
    graph.add_node("shape_extract", traced_node("shape_extract")(_shape_extract))
    graph.add_node("dtype_extract", traced_node("dtype_extract")(_dtype_extract))
    graph.add_node("dformat_extract", traced_node("dformat_extract")(_dformat_extract))
    graph.add_node("optional_extract", traced_node("optional_extract")(_optional_extract))
    graph.add_node("array_length_extract", traced_node("array_length_extract")(_array_length_extract))
    graph.add_node("allowed_range_extract", traced_node("allowed_range_extract")(_allowed_range_extract))
    graph.add_node("return_code_extract", traced_node("return_code_extract")(_return_code_extract))
    graph.add_node("determinism_extract", traced_node("determinism_extract")(_determinism_extract))
    graph.add_node("dtype_combo_extract", traced_node("dtype_combo_extract")(_dtype_combo_extract))
    graph.add_node("param_relation_extract", traced_subgraph("param_relation_extract")(create_param_relation_subgraph()))
    graph.add_node("build_param_relations", traced_node("build_param_relations")(_build_param_relations))
    graph.add_node("build_single_param_constraint", traced_node("build_single_param_constraint")(_build_single_param_constraint))
    bpc_subgraph = create_build_param_constraint_subgraph()
    graph.add_node("build_param_constraint", traced_subgraph("build_param_constraint")(bpc_subgraph))
    graph.add_node("assemble_result", traced_node("assemble_result")(_assemble_result))

    if is_first:
        graph.add_edge(START, "init_doc")
        graph.add_conditional_edges("init_doc", _should_continue)

    graph.add_edge("product_support", "table_column_extract")
    graph.add_edge("function_signature_extract", "table_column_extract")
    graph.add_edge("function_explanation_extract", "table_column_extract")
    graph.add_edge("table_column_extract", "llm_description_extract")

    # Fan out: detail extractors run in parallel after llm_description_extract
    for n in _DETAIL_EXTRACTORS:
        graph.add_edge("llm_description_extract", n)

    # Converge: param_relation_extract goes to build_param_relations,
    # others go to build_single_param_constraint
    for n in _DETAIL_EXTRACTORS:
        if n == "param_relation_extract":
            graph.add_edge(n, "build_param_relations")
        else:
            graph.add_edge(n, "build_single_param_constraint")

    graph.add_edge("build_param_relations", "build_single_param_constraint")
    graph.add_edge("build_single_param_constraint", "build_param_constraint")
    graph.add_edge("build_param_constraint", "assemble_result")

    if is_last:
        graph.add_edge("assemble_result", END)


def _build_generate(graph: StateGraph, *, is_first: bool, is_last: bool) -> None:
    for name, fn in [
        ("case_match_model", _case_match_model),
        ("case_init_static", _case_init_static),
        ("case_solve_constraints", _case_solve_constraints),
        ("case_generate", _case_generate),
    ]:
        graph.add_node(name, traced_node(name)(fn))

    if is_first:
        graph.add_edge(START, "case_match_model")

    graph.add_conditional_edges("case_match_model", _route_after_case_match)
    graph.add_edge("case_init_static", "case_solve_constraints")
    graph.add_conditional_edges("case_init_static", _route_after_case_init)
    graph.add_edge("case_solve_constraints", "case_generate")

    if is_last:
        graph.add_edge("case_generate", END)


def _build_execute(graph: StateGraph, *, is_first: bool, is_last: bool) -> None:
    for name, fn in [
        ("exec_generate_atk", _exec_generate_atk),
        ("exec_cpu_derivation", _exec_cpu_derivation),
        ("exec_run_atk", _exec_run_atk),
    ]:
        graph.add_node(name, traced_node(name)(fn))

    if is_first:
        graph.add_edge(START, "exec_generate_atk")

    graph.add_conditional_edges("exec_generate_atk", _route_after_exec_generate)
    graph.add_conditional_edges("exec_cpu_derivation", _route_after_exec_derivation)
    graph.add_edge("exec_run_atk", END)


# -------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------

_STAGE_BUILDERS = {
    PipelineStage.EXTRACT: _build_extract,
    PipelineStage.GENERATE: _build_generate,
    PipelineStage.EXECUTE: _build_execute,
}


def build_pipeline(stages: list[PipelineStage | str]) -> CompiledStateGraph:
    """Build a pipeline graph from the requested stages."""
    if not stages:
        raise ValueError("stages must not be empty")

    normalised: list[PipelineStage] = []
    for s in stages:
        if isinstance(s, PipelineStage):
            normalised.append(s)
        else:
            try:
                normalised.append(PipelineStage(s))
            except ValueError:
                raise ValueError(
                    f"Invalid stage {s!r}. Must be one of: "
                    f"{[e.value for e in PipelineStage]}"
                ) from None

    order_indices = [_STAGE_ORDER.index(s) for s in normalised]
    if order_indices != sorted(order_indices):
        raise ValueError(
            f"Stages must follow canonical order "
            f"(extract -> generate -> execute), got: {[s.value for s in normalised]}"
        )

    seen: set[PipelineStage] = set()
    unique: list[PipelineStage] = []
    for s in normalised:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    graph = StateGraph(PipelineState)

    for idx, stage in enumerate(unique):
        is_first = (idx == 0)
        is_last = (idx == len(unique) - 1)
        _STAGE_BUILDERS[stage](graph, is_first=is_first, is_last=is_last)

        if not is_last:
            nxt = unique[idx + 1]
            graph.add_edge(
                _STAGE_LAST_NODE[stage],
                _STAGE_FIRST_NODE[nxt],
            )

    label = "+".join(s.value for s in unique)
    return graph.compile(name=f"pipeline-{label}")
