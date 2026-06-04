"""LangGraph pipeline graph for operator document processing.

Provides a deterministic pipeline graph for structured processing:
InitDoc → [ProductSupport ∥ ParseParams ∥ FunctionSignatureExtract
          ∥ FunctionExplanationExtract]
       → SrcContentExtract → ParamDescExtract
       → [ShapeExtract ∥ DtypeExtract ∥ DFormatExtract ∥ OptionalExtract
          ∥ ParamAttrExtract ∥ ArrayLengthExtract ∥ AllowedRangeExtract
          ∥ ParamRelationExtract]
       → BuildParamConstraint → AssembleResult → END
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.allowed_range_extract import allowed_range_extract_node as _allowed_range_extract
from agent.nodes.array_length_extract import array_length_extract_node as _array_length_extract
from agent.nodes.assemble_result import assemble_result_node as _assemble_result
from agent.nodes.build_param_constraint import build_param_constraint_node as _build_param_constraint
from agent.nodes.determinism_extract import determinism_extract_node as _determinism_extract
from agent.nodes.dformat_extract import dformat_extract_node as _dformat_extract
from agent.nodes.dtype_combo_extract import dtype_combo_extract_node as _dtype_combo_extract
from agent.nodes.dtype_extract import dtype_extract_node as _dtype_extract
from agent.nodes.function_signature_extract import function_signature_extract_node as _function_signature_extract
from agent.nodes.function_explanation_extract import (
    function_explanation_extract_node as _function_explanation_extract,
)
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


def _should_continue(state: dict) -> list[str]:
    """Route after init_doc: fan out to four nodes, or END on error."""
    status = state.get("status", "new")
    if status in ("error",):
        return END
    return ["product_support", "parse_params", "function_signature_extract", "function_explanation_extract"]


def create_pipeline_graph() -> CompiledStateGraph:
    """Build a deterministic pipeline for document processing.

    Nodes are wrapped with @traced_node for automatic span + event emission.

    Flow:
    InitDoc → [error → END |
               ProductSupport ∥ ParseParams ∥ FunctionSignatureExtract
               ∥ FunctionExplanationExtract →
               SrcContentExtract →
                ParamDescExtract → [ShapeExtract ∥ DtypeExtract ∥ DFormatExtract
                                    ∥ OptionalExtract ∥ ParamAttrExtract
                                    ∥ ArrayLengthExtract ∥ AllowedRangeExtract
                                    ∥ ParamRelationExtract]
               → BuildParamConstraint → AssembleResult → END]
    """
    graph = StateGraph(PipelineState)
    graph.add_node("init_doc", traced_node("init_doc")(_init_doc))
    graph.add_node("product_support", traced_node("product_support")(_product_support))
    graph.add_node("parse_params", traced_node("parse_params")(_parse_params))
    graph.add_node("src_content_extract", traced_node("src_content_extract")(_src_content_extract))
    graph.add_node("function_signature_extract", traced_node("function_signature_extract")(_function_signature_extract))
    graph.add_node("function_explanation_extract", traced_node("function_explanation_extract")(_function_explanation_extract))
    graph.add_node("param_desc_extract", traced_node("param_desc_extract")(_param_desc_extract))
    graph.add_node("shape_extract", traced_node("shape_extract")(_shape_extract))
    graph.add_node("dtype_extract", traced_node("dtype_extract")(_dtype_extract))
    graph.add_node("dformat_extract", traced_node("dformat_extract")(_dformat_extract))
    graph.add_node("optional_extract", traced_node("optional_extract")(_optional_extract))
    graph.add_node("param_attr_extract", traced_node("param_attr_extract")(_param_attr_extract))
    graph.add_node("array_length_extract", traced_node("array_length_extract")(_array_length_extract))
    graph.add_node("allowed_range_extract", traced_node("allowed_range_extract")(_allowed_range_extract))
    graph.add_node("return_code_extract", traced_node("return_code_extract")(_return_code_extract))
    graph.add_node("determinism_extract", traced_node("determinism_extract")(_determinism_extract))
    graph.add_node("dtype_combo_extract", traced_node("dtype_combo_extract")(_dtype_combo_extract))
    graph.add_node("build_param_constraint", traced_node("build_param_constraint")(_build_param_constraint))
    graph.add_node("assemble_result", traced_node("assemble_result")(_assemble_result))

    param_relation_subgraph = create_param_relation_subgraph()
    graph.add_node("param_relation_extract", traced_node("param_relation_extract")(param_relation_subgraph))

    graph.add_edge(START, "init_doc")
    graph.add_conditional_edges("init_doc", _should_continue)
    graph.add_edge("product_support", "src_content_extract")
    graph.add_edge("parse_params", "src_content_extract")
    graph.add_edge("function_signature_extract", "src_content_extract")
    graph.add_edge("function_explanation_extract", "src_content_extract")
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
    graph.add_edge("shape_extract", "build_param_constraint")
    graph.add_edge("dtype_extract", "build_param_constraint")
    graph.add_edge("dformat_extract", "build_param_constraint")
    graph.add_edge("optional_extract", "build_param_constraint")
    graph.add_edge("param_attr_extract", "build_param_constraint")
    graph.add_edge("array_length_extract", "build_param_constraint")
    graph.add_edge("allowed_range_extract", "build_param_constraint")
    graph.add_edge("return_code_extract", "build_param_constraint")
    graph.add_edge("determinism_extract", "build_param_constraint")
    graph.add_edge("dtype_combo_extract", "build_param_constraint")
    graph.add_edge("param_relation_extract", "build_param_constraint")
    graph.add_edge("build_param_constraint", "assemble_result")
    graph.add_edge("assemble_result", END)
    return graph.compile(name="operator-pipeline")


def create_pipeline_graph_after_init() -> CompiledStateGraph:
    """Build pipeline starting from fan-out after init_doc has run externally.

    Used when init_doc is executed separately so doc_id can be persisted
    to pipeline_runs immediately rather than waiting for the full pipeline.
    Graph: START → [product_support ∥ parse_params ∥ function_signature_extract
                    ∥ function_explanation_extract] → src_content_extract →
    param_desc_extract → [shape_extract ∥ dtype_extract ∥ dformat_extract
                          ∥ optional_extract ∥ param_attr_extract
                          ∥ array_length_extract ∥ allowed_range_extract
                          ∥ return_code_extract ∥ determinism_extract
                          ∥ dtype_combo_extract ∥ param_relation_extract]
    → build_param_constraint → assemble_result → END
    """
    graph = StateGraph(PipelineState)
    graph.add_node("product_support", traced_node("product_support")(_product_support))
    graph.add_node("parse_params", traced_node("parse_params")(_parse_params))
    graph.add_node("function_signature_extract", traced_node("function_signature_extract")(_function_signature_extract))
    graph.add_node("function_explanation_extract", traced_node("function_explanation_extract")(_function_explanation_extract))
    graph.add_node("src_content_extract", traced_node("src_content_extract")(_src_content_extract))
    graph.add_node("param_desc_extract", traced_node("param_desc_extract")(_param_desc_extract))
    graph.add_node("shape_extract", traced_node("shape_extract")(_shape_extract))
    graph.add_node("dtype_extract", traced_node("dtype_extract")(_dtype_extract))
    graph.add_node("dformat_extract", traced_node("dformat_extract")(_dformat_extract))
    graph.add_node("optional_extract", traced_node("optional_extract")(_optional_extract))
    graph.add_node("param_attr_extract", traced_node("param_attr_extract")(_param_attr_extract))
    graph.add_node("array_length_extract", traced_node("array_length_extract")(_array_length_extract))
    graph.add_node("allowed_range_extract", traced_node("allowed_range_extract")(_allowed_range_extract))
    graph.add_node("return_code_extract", traced_node("return_code_extract")(_return_code_extract))
    graph.add_node("determinism_extract", traced_node("determinism_extract")(_determinism_extract))
    graph.add_node("dtype_combo_extract", traced_node("dtype_combo_extract")(_dtype_combo_extract))
    graph.add_node("build_param_constraint", traced_node("build_param_constraint")(_build_param_constraint))
    graph.add_node("assemble_result", traced_node("assemble_result")(_assemble_result))

    param_relation_subgraph = create_param_relation_subgraph()
    graph.add_node("param_relation_extract", traced_node("param_relation_extract")(param_relation_subgraph))

    graph.add_edge(START, "product_support")
    graph.add_edge(START, "parse_params")
    graph.add_edge(START, "function_signature_extract")
    graph.add_edge(START, "function_explanation_extract")
    graph.add_edge("product_support", "src_content_extract")
    graph.add_edge("parse_params", "src_content_extract")
    graph.add_edge("function_signature_extract", "src_content_extract")
    graph.add_edge("function_explanation_extract", "src_content_extract")
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
    graph.add_edge("shape_extract", "build_param_constraint")
    graph.add_edge("dtype_extract", "build_param_constraint")
    graph.add_edge("dformat_extract", "build_param_constraint")
    graph.add_edge("optional_extract", "build_param_constraint")
    graph.add_edge("param_attr_extract", "build_param_constraint")
    graph.add_edge("array_length_extract", "build_param_constraint")
    graph.add_edge("allowed_range_extract", "build_param_constraint")
    graph.add_edge("return_code_extract", "build_param_constraint")
    graph.add_edge("determinism_extract", "build_param_constraint")
    graph.add_edge("dtype_combo_extract", "build_param_constraint")
    graph.add_edge("param_relation_extract", "build_param_constraint")
    graph.add_edge("build_param_constraint", "assemble_result")
    graph.add_edge("assemble_result", END)
    return graph.compile(name="operator-pipeline")
