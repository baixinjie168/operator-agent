"""LangGraph pipeline graph for operator document processing.

Provides a deterministic pipeline graph for structured processing:
InitDoc → [ProductSupport ∥ FunctionSignatureExtract
          ∥ FunctionExplanationExtract]
       → TableColumnExtract
       → LlmDescriptionExtract
       → [ShapeExtract ∥ DtypeExtract ∥ DFormatExtract ∥ OptionalExtract
          ∥ ArrayLengthExtract ∥ AllowedRangeExtract
          ∥ ParamRelationExtract (subgraph: implicit_param_extract
             + parameter_representation_build + Agent loop)
          ∥ ReturnCodeExtract
          ∥ DeterminismExtract ∥ DtypeComboExtract]
       → BuildParamRelations
       → BuildSingleParamConstraint
       → BuildParamConstraint (subgraph) → AssembleResult → END

Note: FunctionSignatureExtract also produces the flat parameters list
(replaces the old parse_params node)。
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.llm_description_extract import create_description_extract_subgraph
from agent.nodes.allowed_range_extract import allowed_range_extract_node
from agent.nodes.array_length_extract import array_length_extract_node
from agent.nodes.assemble_result import assemble_result_node
from agent.nodes.build_param_constraint import (
    create_build_param_constraint_subgraph,
)
from agent.nodes.build_param_relations import build_param_relations_node
from agent.nodes.single_param_constraint import (
    build_single_param_constraint_node,
)
from agent.nodes.determinism_extract import determinism_extract_node
from agent.nodes.dformat_extract import dformat_extract_node
from agent.nodes.dtype_combo_extract import dtype_combo_extract_node
from agent.nodes.dtype_extract import dtype_extract_node
from agent.nodes.function_signature_extract import function_signature_extract_node
from agent.nodes.function_explanation_extract import (
    function_explanation_extract_node,
)
from agent.nodes.init_doc import init_doc_node
from agent.nodes.optional_extract import optional_extract_node
from agent.nodes.param_relation_extract import create_param_relation_subgraph
from agent.nodes.product_support import product_support_node
from agent.nodes.return_code_extract import return_code_extract_node
from agent.nodes.shape_extract import shape_extract_node
from agent.nodes.state import PipelineState
from agent.nodes.table_column_extract import table_column_extract_node

logger = logging.getLogger(__name__)


def _should_continue(state: dict) -> list[str]:
    """Route after init_doc: fan out to three nodes, or END on error."""
    status = state.get("status", "new")
    if status in ("error",):
        return END
    return ["product_support", "function_signature_extract", "function_explanation_extract"]


def create_pipeline_graph() -> CompiledStateGraph:
    """Build a deterministic pipeline for document processing.

    Flow:
    InitDoc → [error → END |
               ProductSupport ∥ FunctionSignatureExtract (also produces parameters)
               ∥ FunctionExplanationExtract →
               TableColumnExtract →
               LlmDescriptionExtract →
                [ShapeExtract ∥ DtypeExtract ∥ DFormatExtract
                                    ∥ OptionalExtract
                                    ∥ ArrayLengthExtract ∥ AllowedRangeExtract
                                    ∥ ReturnCodeExtract ∥ DeterminismExtract
                                    ∥ DtypeComboExtract ∥ ParamRelationExtract
                                       (subgraph: implicit_param_extract
                                        + parameter_representation_build
                                        + Agent loop)]
               → BuildParamRelations → BuildSingleParamConstraint
               → BuildParamConstraint (subgraph)
               → AssembleResult → END]

    Returns a LangGraph ``CompiledStateGraph`` using ``PipelineState``.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("init_doc", init_doc_node)
    graph.add_node("product_support", product_support_node)
    graph.add_node("table_column_extract", table_column_extract_node)
    description_subgraph = create_description_extract_subgraph()
    graph.add_node("llm_description_extract", description_subgraph)
    graph.add_node("function_signature_extract", function_signature_extract_node)
    graph.add_node("function_explanation_extract", function_explanation_extract_node)
    graph.add_node("shape_extract", shape_extract_node)
    graph.add_node("dtype_extract", dtype_extract_node)
    graph.add_node("dformat_extract", dformat_extract_node)
    graph.add_node("optional_extract", optional_extract_node)
    graph.add_node("array_length_extract", array_length_extract_node)
    graph.add_node("allowed_range_extract", allowed_range_extract_node)
    graph.add_node("return_code_extract", return_code_extract_node)
    graph.add_node("determinism_extract", determinism_extract_node)
    graph.add_node("dtype_combo_extract", dtype_combo_extract_node)
    bpc_subgraph = create_build_param_constraint_subgraph()
    graph.add_node("build_param_constraint", bpc_subgraph)
    graph.add_node("build_param_relations", build_param_relations_node)
    graph.add_node(
        "build_single_param_constraint",
        build_single_param_constraint_node,
    )
    graph.add_node("assemble_result", assemble_result_node)

    param_relation_subgraph = create_param_relation_subgraph()
    graph.add_node("param_relation_extract", param_relation_subgraph)

    graph.add_edge(START, "init_doc")
    graph.add_conditional_edges("init_doc", _should_continue)
    graph.add_edge("product_support", "table_column_extract")
    graph.add_edge("function_signature_extract", "table_column_extract")
    graph.add_edge("function_explanation_extract", "table_column_extract")
    graph.add_edge("table_column_extract", "llm_description_extract")
    graph.add_edge("llm_description_extract", "shape_extract")
    graph.add_edge("llm_description_extract", "dtype_extract")
    graph.add_edge("llm_description_extract", "dformat_extract")
    graph.add_edge("llm_description_extract", "optional_extract")
    graph.add_edge("llm_description_extract", "array_length_extract")
    graph.add_edge("llm_description_extract", "allowed_range_extract")
    graph.add_edge("llm_description_extract", "return_code_extract")
    graph.add_edge("llm_description_extract", "determinism_extract")
    graph.add_edge("llm_description_extract", "dtype_combo_extract")
    graph.add_edge("llm_description_extract", "param_relation_extract")
    graph.add_edge("shape_extract", "build_param_constraint")
    graph.add_edge("dtype_extract", "build_param_constraint")
    graph.add_edge("dformat_extract", "build_param_constraint")
    graph.add_edge("optional_extract", "build_param_constraint")
    graph.add_edge("array_length_extract", "build_param_constraint")
    graph.add_edge("allowed_range_extract", "build_param_constraint")
    graph.add_edge("return_code_extract", "build_param_constraint")
    graph.add_edge("determinism_extract", "build_param_constraint")
    graph.add_edge("dtype_combo_extract", "build_param_constraint")
    graph.add_edge("param_relation_extract", "build_param_relations")
    graph.add_edge("build_param_relations", "build_single_param_constraint")
    graph.add_edge(
        "build_single_param_constraint", "build_param_constraint",
    )
    graph.add_edge("build_param_constraint", "assemble_result")
    graph.add_edge("assemble_result", END)
    return graph.compile(name="operator-pipeline")
