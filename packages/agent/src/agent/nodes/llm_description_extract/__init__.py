"""LlmDescriptionExtract sub-graph: generate logic-complete parameter descriptions via LLM.

Replaces the old monolithic ``llm_description_extract.py`` with a LangGraph
sub-graph that mirrors the ``param_relation_extract`` sub-graph structure.

Flow:
    START -> fetch_sections -> [extract_ws || extract_exe] -> validate_results -> verify_and_enhance -> save_descriptions -> END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.llm_description_extract.extract_exe import extract_exe_node
from agent.nodes.llm_description_extract.extract_ws import extract_ws_node
from agent.nodes.llm_description_extract.fetch_sections import fetch_sections_node
from agent.nodes.llm_description_extract.save_descriptions import save_descriptions_node
from agent.nodes.llm_description_extract.state import DescriptionExtractState
from agent.nodes.llm_description_extract.validate_results import validate_results_node
from agent.nodes.llm_description_extract.verify_and_enhance import verify_and_enhance_node


def create_description_extract_subgraph() -> CompiledStateGraph:
    """Build the llm_description extraction sub-graph.

    Flow:
        fetch_sections -> [extract_ws || extract_exe] -> validate_results
            -> verify_and_enhance -> save_descriptions
    """
    graph = StateGraph(DescriptionExtractState)
    graph.add_node("fetch_sections", fetch_sections_node)
    graph.add_node("extract_ws", extract_ws_node)
    graph.add_node("extract_exe", extract_exe_node)
    graph.add_node("validate_results", validate_results_node)
    graph.add_node("verify_and_enhance", verify_and_enhance_node)
    graph.add_node("save_descriptions", save_descriptions_node)

    graph.add_edge(START, "fetch_sections")
    graph.add_edge("fetch_sections", "extract_ws")
    graph.add_edge("fetch_sections", "extract_exe")
    graph.add_edge("extract_ws", "validate_results")
    graph.add_edge("extract_exe", "validate_results")
    graph.add_edge("validate_results", "verify_and_enhance")
    graph.add_edge("verify_and_enhance", "save_descriptions")
    graph.add_edge("save_descriptions", END)

    return graph.compile(name="llm-description-extract")

