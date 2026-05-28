"""LangGraph pipeline graph for operator document processing.

Provides a deterministic pipeline graph for structured processing:
InitDoc → [ProductSupport ∥ ParseParams] → ParamDescExtract → ShapeExtract → END
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.init_doc import init_doc_node
from agent.nodes.param_desc_extract import param_desc_extract_node
from agent.nodes.parse_params import parse_params_node
from agent.nodes.product_support import product_support_node
from agent.nodes.shape_extract import shape_extract_node
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)


def _should_continue(state: dict) -> list[str]:
    """Route after init_doc: fan out to both nodes, or END on error."""
    status = state.get("status", "new")
    if status in ("error",):
        return END
    return ["product_support", "parse_params"]


def create_pipeline_graph() -> CompiledStateGraph:
    """Build a deterministic pipeline for document processing.

    Flow:
    InitDoc → [error → END |
               ProductSupport ∥ ParseParams → ParamDescExtract → ShapeExtract → END]

    Returns a LangGraph ``CompiledStateGraph`` using ``PipelineState``.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("init_doc", init_doc_node)
    graph.add_node("product_support", product_support_node)
    graph.add_node("parse_params", parse_params_node)
    graph.add_node("param_desc_extract", param_desc_extract_node)
    graph.add_node("shape_extract", shape_extract_node)
    graph.add_edge(START, "init_doc")
    graph.add_conditional_edges("init_doc", _should_continue)
    graph.add_edge("product_support", "param_desc_extract")
    graph.add_edge("parse_params", "param_desc_extract")
    graph.add_edge("param_desc_extract", "shape_extract")
    graph.add_edge("shape_extract", END)
    return graph.compile(name="operator-pipeline")
