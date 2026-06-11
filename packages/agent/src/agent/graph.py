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
"""

from __future__ import annotations

import logging
from enum import Enum

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.allowed_range_extract import allowed_range_extract_node as _allowed_range_extract
from agent.nodes.array_length_extract import array_length_extract_node as _array_length_extract
from agent.nodes.assemble_result import assemble_result_node as _assemble_result
from agent.nodes.build_param_constraint import build_param_constraint_node as _build_param_constraint
from agent.nodes.build_param_relations import build_param_relations_node as _build_param_relations
from agent.nodes.case_subgraph import (
    case_generate_node as _case_generate,
    case_init_static_node as _case_init_static,
    case_match_model_node as _case_match_model,
    case_solve_constraints_node as _case_solve_constraints,
)
from agent.nodes.determinism_extract import determinism_extract_node as _determinism_extract
from agent.nodes.dformat_extract import dformat_extract_node as _dformat_extract
from agent.nodes.dtype_combo_extract import dtype_combo_extract_node as _dtype_combo_extract
from agent.nodes.dtype_extract import dtype_extract_node as _dtype_extract
from agent.nodes.executer_subgraph import (
    exec_cpu_derivation_node as _exec_cpu_derivation,
    exec_generate_atk_node as _exec_generate_atk,
    exec_run_atk_node as _exec_run_atk,
)
from agent.nodes.function_explanation_extract import (
    function_explanation_extract_node as _function_explanation_extract,
)
from agent.nodes.function_signature_extract import function_signature_extract_node as _function_signature_extract
from agent.nodes.init_doc import init_doc_node as _init_doc
from agent.nodes.optional_extract import optional_extract_node as _optional_extract
from agent.nodes.param_attr_extract import param_attr_extract_node as _param_attr_extract
from agent.nodes.param_desc_extract import param_desc_extract_node as _param_desc_extract
from agent.nodes.param_relation_extract import create_param_relation_subgraph
from agent.nodes.parse_params import parse_params_node as _parse_params
from agent.nodes.product_support import product_support_node as _product_support
from agent.nodes.return_code_extract import return_code_extract_node as _return_code_extract
from agent.nodes.shape_extract import shape_extract_node as _shape_extract
from agent.nodes.src_content_extract import src_content_extract_node as _src_content_extract
from agent.nodes.state import PipelineState
from agent.runtime import traced_node

logger = logging.getLogger(__name__)


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

def _route_after_case_match(state: dict) -> str:
    return END if state.get("error") else "case_init_static"


def _route_after_case_init(state: dict) -> str:
    return END if state.get("error") else "case_solve_constraints"


def _route_after_exec_generate(state: dict) -> str:
    return END if state.get("error") else "exec_cpu_derivation"


def _route_after_exec_derivation(state: dict) -> str:
    return END if state.get("error") else "exec_run_atk"


# -------------------------------------------------------------------
#  Node lists
# -------------------------------------------------------------------

_PARALLEL_FAN_OUT = [
    "product_support",
    "parse_params",
    "function_signature_extract",
    "function_explanation_extract",
]

_EXTRACT_NODES = [
    ("product_support", _product_support),
    ("parse_params", _parse_params),
    ("function_signature_extract", _function_signature_extract),
    ("function_explanation_extract", _function_explanation_extract),
    ("src_content_extract", _src_content_extract),
    ("param_desc_extract", _param_desc_extract),
    ("shape_extract", _shape_extract),
    ("dtype_extract", _dtype_extract),
    ("dformat_extract", _dformat_extract),
    ("optional_extract", _optional_extract),
    ("param_attr_extract", _param_attr_extract),
    ("array_length_extract", _array_length_extract),
    ("allowed_range_extract", _allowed_range_extract),
    ("return_code_extract", _return_code_extract),
    ("determinism_extract", _determinism_extract),
    ("dtype_combo_extract", _dtype_combo_extract),
    ("build_param_relations", _build_param_relations),
    ("build_param_constraint", _build_param_constraint),
    ("assemble_result", _assemble_result),
]

_GENERATE_NODES = [
    ("case_match_model", _case_match_model),
    ("case_init_static", _case_init_static),
    ("case_solve_constraints", _case_solve_constraints),
    ("case_generate", _case_generate),
]

_EXECUTE_NODES = [
    ("exec_generate_atk", _exec_generate_atk),
    ("exec_cpu_derivation", _exec_cpu_derivation),
    ("exec_run_atk", _exec_run_atk),
]

_DETAIL_EXTRACTORS = [
    "shape_extract", "dtype_extract", "dformat_extract",
    "optional_extract", "param_attr_extract", "array_length_extract",
    "allowed_range_extract", "return_code_extract", "determinism_extract",
    "dtype_combo_extract", "param_relation_extract",
]

_STAGE_ORDER = [PipelineStage.EXTRACT, PipelineStage.GENERATE, PipelineStage.EXECUTE]

_STAGE_LAST_NODE = {
    PipelineStage.EXTRACT: "assemble_result",
    PipelineStage.GENERATE: "case_generate",
    PipelineStage.EXECUTE: "exec_run_atk",
}

_STAGE_FIRST_NODE = {
    PipelineStage.EXTRACT: None,
    PipelineStage.GENERATE: "case_match_model",
    PipelineStage.EXECUTE: "exec_generate_atk",
}


# -------------------------------------------------------------------
#  Stage builders
# -------------------------------------------------------------------

def _build_extract(graph: StateGraph, *, is_first: bool, is_last: bool) -> None:
    for name, fn in _EXTRACT_NODES:
        graph.add_node(name, traced_node(name)(fn))
    graph.add_node("param_relation_extract", create_param_relation_subgraph())

    if is_first:
        for n in _PARALLEL_FAN_OUT:
            graph.add_edge(START, n)

    for n in _PARALLEL_FAN_OUT:
        graph.add_edge(n, "src_content_extract")

    graph.add_edge("src_content_extract", "param_desc_extract")

    for n in _DETAIL_EXTRACTORS:
        graph.add_edge("param_desc_extract", n)

    for n in _DETAIL_EXTRACTORS:
        if n == "param_relation_extract":
            graph.add_edge(n, "build_param_relations")
        else:
            graph.add_edge(n, "build_param_constraint")

    graph.add_edge("build_param_relations", "build_param_constraint")
    graph.add_edge("build_param_constraint", "assemble_result")

    if is_last:
        graph.add_edge("assemble_result", END)


def _build_generate(graph: StateGraph, *, is_first: bool, is_last: bool) -> None:
    for name, fn in _GENERATE_NODES:
        graph.add_node(name, traced_node(name)(fn))

    if is_first:
        graph.add_edge(START, "case_match_model")

    # Error short-circuits at match_model and init_static
    graph.add_conditional_edges("case_match_model", _route_after_case_match)
    graph.add_edge("case_init_static", "case_solve_constraints")
    graph.add_conditional_edges("case_init_static", _route_after_case_init)
    # solve_constraints -> generate (unconditional; generate handles its own errors)
    graph.add_edge("case_solve_constraints", "case_generate")


def _build_execute(graph: StateGraph, *, is_first: bool, is_last: bool) -> None:
    for name, fn in _EXECUTE_NODES:
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
    """Build a pipeline graph from the requested stages.

    Args:
        stages: Ordered list of stages. Accepts PipelineStage values or
            plain strings ("extract", "generate", "execute").
            Must follow canonical order: extract -> generate -> execute.

    Returns:
        A compiled CompiledStateGraph ready for ainvoke.

    Raises:
        ValueError: If stages is empty or contains invalid values.
    """
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
