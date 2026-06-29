"""BuildParamConstraint sub-graph: assemble structured param_constraint JSON.

Simplified: cross_reference_resolve (value copy moved to fetch_param_data)
and allowed_range_build (moved to constraint_extract Pass 4) have been removed.

Flow:
    START -> fetch_param_data
               |
     +---------+------------+
     v                      v
  dimensions_build        attrs_build
     |                      |
     +----------+-----------+
                v
       constraint_assemble
                |
                v
               END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.build_param_constraint.attrs_build import attrs_build_node
from agent.nodes.build_param_constraint.constraint_assemble import (
    constraint_assemble_node,
)
from agent.nodes.build_param_constraint.dimensions_agent import dimensions_agent_node
from agent.nodes.build_param_constraint.fetch_param_data import fetch_param_data_node
from agent.nodes.build_param_constraint.state import BuildParamConstraintState


def create_build_param_constraint_subgraph() -> CompiledStateGraph:
    """Build the build_param_constraint sub-graph."""
    graph = StateGraph(BuildParamConstraintState)

    graph.add_node("fetch_param_data", fetch_param_data_node)
    graph.add_node("dimensions_build", dimensions_agent_node)
    graph.add_node("attrs_build", attrs_build_node)
    graph.add_node("constraint_assemble", constraint_assemble_node)

    graph.add_edge(START, "fetch_param_data")

    # Two parallel nodes (depend on fetch_param_data completing)
    graph.add_edge("fetch_param_data", "dimensions_build")
    graph.add_edge("fetch_param_data", "attrs_build")

    # Merge into constraint_assemble
    graph.add_edge("dimensions_build", "constraint_assemble")
    graph.add_edge("attrs_build", "constraint_assemble")

    graph.add_edge("constraint_assemble", END)

    return graph.compile(name="build-param-constraint")
