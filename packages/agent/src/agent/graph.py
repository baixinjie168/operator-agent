"""LangGraph pipeline graph for operator document processing.

Provides a deterministic pipeline graph for structured processing:
InitDoc → [ProductSupport ∥ ParseParams] → SrcContentExtract → ParamDescExtract
       → [ShapeExtract ∥ DtypeExtract ∥ OptionalExtract] → END
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.dtype_extract import dtype_extract_node as _dtype_extract
from agent.nodes.init_doc import init_doc_node as _init_doc
from agent.nodes.optional_extract import optional_extract_node as _optional_extract
from agent.nodes.param_desc_extract import param_desc_extract_node as _param_desc_extract
from agent.nodes.parse_params import parse_params_node as _parse_params
from agent.nodes.product_support import product_support_node as _product_support
from agent.nodes.shape_extract import shape_extract_node as _shape_extract
from agent.nodes.src_content_extract import src_content_extract_node as _src_content_extract
from agent.nodes.state import PipelineState
from agent.runtime import traced_node

logger = logging.getLogger(__name__)


def _should_continue(state: dict) -> list[str]:
    """Route after init_doc: fan out to both nodes, or END on error."""
    status = state.get("status", "new")
    if status in ("error",):
        return END
    return ["product_support", "parse_params"]


def create_pipeline_graph() -> CompiledStateGraph:
    """Build a deterministic pipeline for document processing.

    Nodes are wrapped with @traced_node for automatic span + event emission.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("init_doc", traced_node("init_doc")(_init_doc))
    graph.add_node("product_support", traced_node("product_support")(_product_support))
    graph.add_node("parse_params", traced_node("parse_params")(_parse_params))
    graph.add_node("src_content_extract", traced_node("src_content_extract")(_src_content_extract))
    graph.add_node("param_desc_extract", traced_node("param_desc_extract")(_param_desc_extract))
    graph.add_node("shape_extract", traced_node("shape_extract")(_shape_extract))
    graph.add_node("dtype_extract", traced_node("dtype_extract")(_dtype_extract))
    graph.add_node("optional_extract", traced_node("optional_extract")(_optional_extract))
    graph.add_edge(START, "init_doc")
    graph.add_conditional_edges("init_doc", _should_continue)
    graph.add_edge("product_support", "src_content_extract")
    graph.add_edge("parse_params", "src_content_extract")
    graph.add_edge("src_content_extract", "param_desc_extract")
    graph.add_edge("param_desc_extract", "shape_extract")
    graph.add_edge("param_desc_extract", "dtype_extract")
    graph.add_edge("param_desc_extract", "optional_extract")
    graph.add_edge("shape_extract", END)
    graph.add_edge("dtype_extract", END)
    graph.add_edge("optional_extract", END)
    return graph.compile(name="operator-pipeline")


def create_pipeline_graph_after_init() -> CompiledStateGraph:
    """Build pipeline starting from fan-out after init_doc has run externally.

    Used when init_doc is executed separately so doc_id can be persisted
    to pipeline_runs immediately rather than waiting for the full pipeline.
    Graph: START → [product_support ∥ parse_params] → src_content_extract →
    param_desc_extract → [shape_extract ∥ dtype_extract ∥ optional_extract] → END
    """
    graph = StateGraph(PipelineState)
    graph.add_node("product_support", traced_node("product_support")(_product_support))
    graph.add_node("parse_params", traced_node("parse_params")(_parse_params))
    graph.add_node("src_content_extract", traced_node("src_content_extract")(_src_content_extract))
    graph.add_node("param_desc_extract", traced_node("param_desc_extract")(_param_desc_extract))
    graph.add_node("shape_extract", traced_node("shape_extract")(_shape_extract))
    graph.add_node("dtype_extract", traced_node("dtype_extract")(_dtype_extract))
    graph.add_node("optional_extract", traced_node("optional_extract")(_optional_extract))
    graph.add_edge(START, "product_support")
    graph.add_edge(START, "parse_params")
    graph.add_edge("product_support", "src_content_extract")
    graph.add_edge("parse_params", "src_content_extract")
    graph.add_edge("src_content_extract", "param_desc_extract")
    graph.add_edge("param_desc_extract", "shape_extract")
    graph.add_edge("param_desc_extract", "dtype_extract")
    graph.add_edge("param_desc_extract", "optional_extract")
    graph.add_edge("shape_extract", END)
    graph.add_edge("dtype_extract", END)
    graph.add_edge("optional_extract", END)
    return graph.compile(name="operator-pipeline")
