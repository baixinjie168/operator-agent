"""BuildParamConstraint sub-graph: assemble structured param_constraint JSON.

Flow:
    START -> fetch_param_data
               |
               v
        cross_reference_resolve   (resolve "与xx一致" refs)
               |
     +---------+------------+
     v         v            v
  dimensions   allowed_range  attrs_build
  _agent       _build         (deterministic)
     |         |            |
     +----+----+            |
          v                  |
   constraint_assemble <----+
          |
          v
         END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.build_param_constraint.allowed_range_build import (
    allowed_range_build_node,
)
from agent.nodes.build_param_constraint.attrs_build import attrs_build_node
from agent.nodes.build_param_constraint.constraint_assemble import (
    constraint_assemble_node,
)
from agent.nodes.build_param_constraint.cross_reference_resolve import (
    cross_reference_resolve_node,
)
from agent.nodes.build_param_constraint.dimensions_agent import dimensions_agent_node
from agent.nodes.build_param_constraint.fetch_param_data import fetch_param_data_node
from agent.nodes.build_param_constraint.state import BuildParamConstraintState


def create_build_param_constraint_subgraph() -> CompiledStateGraph:
    """Build the build_param_constraint sub-graph."""
    graph = StateGraph(BuildParamConstraintState)

    graph.add_node("fetch_param_data", fetch_param_data_node)
    graph.add_node("cross_reference_resolve", cross_reference_resolve_node)
    graph.add_node("dimensions_build", dimensions_agent_node)
    graph.add_node("allowed_range_build", allowed_range_build_node)
    graph.add_node("attrs_build", attrs_build_node)
    graph.add_node("constraint_assemble", constraint_assemble_node)

    graph.add_edge(START, "fetch_param_data")
    graph.add_edge("fetch_param_data", "cross_reference_resolve")

    # Three parallel nodes (depend on cross_reference_resolve completing)
    graph.add_edge("cross_reference_resolve", "dimensions_build")
    graph.add_edge("cross_reference_resolve", "allowed_range_build")
    graph.add_edge("cross_reference_resolve", "attrs_build")

    # Merge into constraint_assemble
    graph.add_edge("dimensions_build", "constraint_assemble")
    graph.add_edge("allowed_range_build", "constraint_assemble")
    graph.add_edge("attrs_build", "constraint_assemble")

    graph.add_edge("constraint_assemble", END)

    return graph.compile(name="build-param-constraint")
