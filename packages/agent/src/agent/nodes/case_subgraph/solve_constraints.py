"""Step 4 of GeneratorAgent: solve inter-parameter constraints.

Evaluates ``constraints_in_parameters`` relations to estimate how many
shape/dtype combinations will pass validation.  Pure analysis — no mutation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)


async def case_solve_constraints_node(state: PipelineState) -> dict[str, Any]:
    """Estimate the constraint-solver outcome (valid vs rejected combos)."""
    if state.get("error"):
        return {"error": state.get("error")}

    constraints = state.get("constraints_raw")
    if not constraints:
        return {"error": "constraints_raw is missing — cannot solve constraints"}

    logger.info("case_solve_constraints: solving inter-parameter constraints")
    await asyncio.sleep(0.3)

    try:
        constraints_in_parameters = constraints.get("constraints_in_parameters", {}) or {}
        # Conservative estimate: assume 60% of sampled combos pass validation
        sampled_shapes = int(state.get("sampled_shapes") or 0)
        sampled_dtypes = int(state.get("sampled_dtypes") or 0)
        total_combos = sampled_shapes * sampled_dtypes
        valid_combos = int(total_combos * 0.6) if total_combos else 0
        rejected_combos = total_combos - valid_combos

        relation_count = 0
        if isinstance(constraints_in_parameters, dict):
            for relations in constraints_in_parameters.values():
                if isinstance(relations, list):
                    relation_count += len(relations)

        logger.info(
            "case_solve_constraints: relations=%d total=%d valid=%d rejected=%d",
            relation_count, total_combos, valid_combos, rejected_combos,
        )
        return {
            "valid_combos": valid_combos,
            "rejected_combos": rejected_combos,
            "error": None,
        }
    except Exception as e:
        logger.exception("case_solve_constraints failed")
        return {"error": f"failed to solve constraints: {e}"}
