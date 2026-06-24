"""GeneratorAgent sub-graph: 4-step case generation pipeline.

Flow (mirrors the GeneratorAgent card in the front-end):
    case_match_model → case_init_static → case_solve_constraints
    → case_generate → END

Each internal node is wrapped with ``@traced_node`` here so spans + SSE events
fire under ``agent_id="case"`` even when this sub-graph is nested inside the
main pipeline graph.

Note: this sub-graph is also a standalone entry point for ``/api/v1/generator/run``.
The state is ``PipelineState`` (shared with the main pipeline) so that the same
case_*_node functions can be reused.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.case_subgraph.generate import case_generate_node
from agent.nodes.case_subgraph.init_static import case_init_static_node
from agent.nodes.case_subgraph.match_model import case_match_model_node
from agent.nodes.case_subgraph.solve_constraints import case_solve_constraints_node
from agent.nodes.state import PipelineState
from agent.runtime import traced_node

__all__ = [
    "case_generate_node",
    "case_init_static_node",
    "case_match_model_node",
    "case_solve_constraints_node",
    "create_case_subgraph",
]


def _should_continue_after_match(state: dict) -> str:
    """Stop after case_match_model on error so subsequent steps don't run."""
    if state.get("error"):
        return END
    return "case_init_static"


def _should_continue_after_init(state: dict) -> str:
    """Stop after case_init_static on error."""
    if state.get("error"):
        return END
    return "case_solve_constraints"


def _should_continue_after_solve(state: dict) -> str:
    """Stop after case_solve_constraints on error."""
    if state.get("error"):
        return END
    return "case_generate"


def create_case_subgraph() -> CompiledStateGraph:
    """Build the 4-step GeneratorAgent sub-graph using :class:`PipelineState`.

    On error, the sub-graph short-circuits to ``END`` so the user gets a clear
    failure in the UI rather than a cascade of misleading step errors.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("case_match_model", traced_node("case_match_model")(case_match_model_node))
    graph.add_node("case_init_static", traced_node("case_init_static")(case_init_static_node))
    graph.add_node("case_solve_constraints", traced_node("case_solve_constraints")(case_solve_constraints_node))
    graph.add_node("case_generate", traced_node("case_generate")(case_generate_node))

    graph.add_edge(START, "case_match_model")
    graph.add_conditional_edges("case_match_model", _should_continue_after_match)
    graph.add_edge("case_init_static", "case_solve_constraints")
    graph.add_conditional_edges("case_init_static", _should_continue_after_init)
    graph.add_edge("case_solve_constraints", "case_generate")
    graph.add_conditional_edges("case_generate", _should_continue_after_solve, {
        "case_generate": "case_generate",
        END: END,
    })
    return graph.compile(name="case-subgraph")
