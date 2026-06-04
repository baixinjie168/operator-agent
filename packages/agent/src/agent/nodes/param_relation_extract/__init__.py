"""ParamRelationExtract sub-graph: extract parameter coupling relations from sections."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.param_relation_extract.extract_relations import (
    extract_exe_node,
    extract_ws_node,
)
from agent.nodes.param_relation_extract.fetch_sections import fetch_sections_node
from agent.nodes.param_relation_extract.merge_relations import merge_relations_node
from agent.nodes.param_relation_extract.save_relations import save_relations_node
from agent.nodes.param_relation_extract.state import RelationExtractState


def create_param_relation_subgraph() -> CompiledStateGraph:
    graph = StateGraph(RelationExtractState)
    graph.add_node("fetch_sections", fetch_sections_node)
    graph.add_node("extract_ws", extract_ws_node)
    graph.add_node("extract_exe", extract_exe_node)
    graph.add_node("merge_relations", merge_relations_node)
    graph.add_node("save_relations", save_relations_node)

    graph.add_edge(START, "fetch_sections")
    graph.add_edge("fetch_sections", "extract_ws")
    graph.add_edge("fetch_sections", "extract_exe")
    graph.add_edge("extract_ws", "merge_relations")
    graph.add_edge("extract_exe", "merge_relations")
    graph.add_edge("merge_relations", "save_relations")
    graph.add_edge("save_relations", END)

    return graph.compile(name="param-relation-extract")
