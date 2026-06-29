"""ParamRelationExtract sub-graph: extract implicit parameters and representations.

Simplified: extract_ws/exe, implicit_value_constraint, merge_relations, and
save_relations have been merged into the constraint_extract node (Pass 3 + 5).
The subgraph now only handles implicit variable extraction and parameter
representation building — both are needed before constraint_extract runs.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.param_relation_extract.fetch_sections import fetch_sections_node
from agent.nodes.param_relation_extract.parameter_representation_build import (
    parameter_representation_build_node,
)
from agent.nodes.param_relation_extract.implicit_param_extract import (
    implicit_param_extract_node,
)
from agent.nodes.param_relation_extract.state import RelationExtractState


def create_param_relation_subgraph() -> CompiledStateGraph:
    graph = StateGraph(RelationExtractState)
    graph.add_node("fetch_sections", fetch_sections_node)
    graph.add_node("implicit_param_extract", implicit_param_extract_node)
    graph.add_node("param_repr_build", parameter_representation_build_node)

    graph.add_edge(START, "fetch_sections")
    graph.add_edge("fetch_sections", "implicit_param_extract")
    graph.add_edge("implicit_param_extract", "param_repr_build")
    graph.add_edge("param_repr_build", END)

    return graph.compile(name="param-relation-extract")
