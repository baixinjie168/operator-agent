"""Step 3 of GeneratorAgent: initialize static data (shapes, dtypes, value ranges).

Counts the shape / dtype / value-range candidates available for sampling.
Pure observation — no mutation.  Result feeds the constraint solver.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)


def _count_shape_candidates(constraints: dict | None) -> int:
    """Count total shape dimension candidates across all inputs."""
    if not constraints:
        return 0
    inputs = constraints.get("inputs", {}) or {}
    if not isinstance(inputs, dict):
        return 0
    total = 0
    for _name, constraint in inputs.items():
        if not isinstance(constraint, dict):
            continue
        dimensions = constraint.get("dimensions") or constraint.get("shape") or []
        if isinstance(dimensions, list):
            total += max(len(dimensions), 1)
    return total


def _count_dtype_candidates(constraints: dict | None) -> int:
    """Count distinct dtype candidates across platforms."""
    if not constraints:
        return 0
    dtype_support = constraints.get("dtype_support_description", {}) or {}
    if not isinstance(dtype_support, dict):
        return 0
    dtypes: set[str] = set()
    for combos in dtype_support.values():
        if not isinstance(combos, list):
            continue
        for combo in combos:
            if isinstance(combo, dict):
                for v in combo.values():
                    if isinstance(v, str):
                        dtypes.add(v)
                    elif isinstance(v, list):
                        dtypes.update(x for x in v if isinstance(x, str))
    return len(dtypes)


async def case_init_static_node(state: PipelineState) -> dict[str, Any]:
    """Initialize static sampling space: count shape & dtype candidates."""
    if state.get("error"):
        return {"error": state.get("error")}

    constraints = state.get("constraints_raw")
    if not constraints:
        return {"error": "constraints_raw is missing — cannot init static data"}

    logger.info("case_init_static: scanning sampling space")
    await asyncio.sleep(0.25)

    try:
        sampled_shapes = _count_shape_candidates(constraints)
        sampled_dtypes = _count_dtype_candidates(constraints)
        return {
            "sampled_shapes": sampled_shapes,
            "sampled_dtypes": sampled_dtypes,
            "error": None,
        }
    except Exception as e:
        logger.exception("case_init_static failed")
        return {"error": f"failed to init static data: {e}"}
