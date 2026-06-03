"""LangGraph pipeline graph for operator document processing.

Provides a deterministic pipeline graph for structured processing:
InitDoc → [ProductSupport ∥ ParseParams ∥ FunctionSignatureExtract]
       → SrcContentExtract → ParamDescExtract
       → [ShapeExtract ∥ DtypeExtract ∥ DFormatExtract ∥ OptionalExtract
          ∥ ParamAttrExtract ∥ ArrayLengthExtract ∥ AllowedRangeExtract
          ∥ ParamRelationExtract] → END
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.allowed_range_extract import allowed_range_extract_node
from agent.nodes.array_length_extract import array_length_extract_node
from agent.nodes.assemble_result import assemble_result_node
from agent.nodes.determinism_extract import determinism_extract_node
from agent.nodes.dformat_extract import dformat_extract_node
from agent.nodes.dtype_combo_extract import dtype_combo_extract_node
from agent.nodes.dtype_extract import dtype_extract_node
from agent.nodes.function_signature_extract import function_signature_extract_node
from agent.nodes.init_doc import init_doc_node
from agent.nodes.optional_extract import optional_extract_node
from agent.nodes.param_attr_extract import param_attr_extract_node
from agent.nodes.param_desc_extract import param_desc_extract_node
from agent.nodes.param_relation_extract import create_param_relation_subgraph
from agent.nodes.parse_params import parse_params_node
from agent.nodes.product_support import product_support_node
from agent.nodes.return_code_extract import return_code_extract_node
from agent.nodes.shape_extract import shape_extract_node
from agent.nodes.src_content_extract import src_content_extract_node
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)


def _should_continue(state: dict) -> list[str]:
    """Route after init_doc: fan out to three nodes, or END on error."""
    status = state.get("status", "new")
    if status in ("error",):
        return END
    return ["product_support", "parse_params", "function_signature_extract"]


def create_pipeline_graph() -> CompiledStateGraph:
    """Build a deterministic pipeline for document processing.

    Flow:
    InitDoc → [error → END |
               ProductSupport ∥ ParseParams ∥ FunctionSignatureExtract →
               SrcContentExtract →
                ParamDescExtract → [ShapeExtract ∥ DtypeExtract ∥ DFormatExtract
                                    ∥ OptionalExtract ∥ ParamAttrExtract
                                    ∥ ArrayLengthExtract ∥ AllowedRangeExtract
                                    ∥ ParamRelationExtract] → END]

    Returns a LangGraph ``CompiledStateGraph`` using ``PipelineState``.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("init_doc", init_doc_node)
    graph.add_node("product_support", product_support_node)
    graph.add_node("parse_params", parse_params_node)
    graph.add_node("src_content_extract", src_content_extract_node)
    graph.add_node("function_signature_extract", function_signature_extract_node)
    graph.add_node("param_desc_extract", param_desc_extract_node)
    graph.add_node("shape_extract", shape_extract_node)
    graph.add_node("dtype_extract", dtype_extract_node)
    graph.add_node("dformat_extract", dformat_extract_node)
    graph.add_node("optional_extract", optional_extract_node)
    graph.add_node("param_attr_extract", param_attr_extract_node)
    graph.add_node("array_length_extract", array_length_extract_node)
    graph.add_node("allowed_range_extract", allowed_range_extract_node)
    graph.add_node("return_code_extract", return_code_extract_node)
    graph.add_node("determinism_extract", determinism_extract_node)
    graph.add_node("dtype_combo_extract", dtype_combo_extract_node)
    graph.add_node("assemble_result", assemble_result_node)

    param_relation_subgraph = create_param_relation_subgraph()
    graph.add_node("param_relation_extract", param_relation_subgraph)

    graph.add_edge(START, "init_doc")
    graph.add_conditional_edges("init_doc", _should_continue)
    graph.add_edge("product_support", "src_content_extract")
    graph.add_edge("parse_params", "src_content_extract")
    graph.add_edge("function_signature_extract", "src_content_extract")
    graph.add_edge("src_content_extract", "param_desc_extract")
    graph.add_edge("param_desc_extract", "shape_extract")
    graph.add_edge("param_desc_extract", "dtype_extract")
    graph.add_edge("param_desc_extract", "dformat_extract")
    graph.add_edge("param_desc_extract", "optional_extract")
    graph.add_edge("param_desc_extract", "param_attr_extract")
    graph.add_edge("param_desc_extract", "array_length_extract")
    graph.add_edge("param_desc_extract", "allowed_range_extract")
    graph.add_edge("param_desc_extract", "return_code_extract")
    graph.add_edge("param_desc_extract", "determinism_extract")
    graph.add_edge("param_desc_extract", "dtype_combo_extract")
    graph.add_edge("param_desc_extract", "param_relation_extract")
    graph.add_edge("shape_extract", "assemble_result")
    graph.add_edge("dtype_extract", "assemble_result")
    graph.add_edge("dformat_extract", "assemble_result")
    graph.add_edge("optional_extract", "assemble_result")
    graph.add_edge("param_attr_extract", "assemble_result")
    graph.add_edge("array_length_extract", "assemble_result")
    graph.add_edge("allowed_range_extract", "assemble_result")
    graph.add_edge("return_code_extract", "assemble_result")
    graph.add_edge("determinism_extract", "assemble_result")
    graph.add_edge("dtype_combo_extract", "assemble_result")
    graph.add_edge("param_relation_extract", "assemble_result")
    graph.add_edge("assemble_result", END)
    return graph.compile(name="operator-pipeline")
