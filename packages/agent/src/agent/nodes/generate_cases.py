"""GenerateCases node: produce ``{operator_name}_cases.json`` from saved constraints.

Reads the ``json_constraints`` field written by ``assemble_result`` via MCP,
runs the pure-Python ``TestCaseGenerator``, and persists the result both to
the ``test_cases`` table and ``cases/{operator_name}_cases.json`` on disk.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.generators import TestCaseGenerator, parse_result_json
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_DEFAULT_COUNT = 10
_DEFAULT_SEED = 42


async def generate_cases_node(state: PipelineState) -> dict[str, Any]:
    """Generate test cases from operator constraints.

    State fields used:
        * ``operator_name`` — drives MCP lookup and output filename.
        * ``cases_count`` — overrides the default generation count.
        * ``cases_seed``  — overrides the default random seed.

    Returns:
        State update with ``error`` (str | None), ``cases_path`` (str | None),
        and ``cases_count`` (int | None).
    """
    operator_name = state.get("operator_name", "")
    count = int(state.get("cases_count") or _DEFAULT_COUNT)
    seed = int(state.get("cases_seed") or _DEFAULT_SEED)

    if not operator_name:
        logger.warning("generate_cases: missing operator_name, skipping")
        return {"error": "operator_name is required", "cases_path": None, "cases_count": None}

    logger.info(
        "generate_cases: starting for %s (count=%d, seed=%d)",
        operator_name, count, seed,
    )

    try:
        constraints = await _mcp_client.get_json_constraints(operator_name)
        if not constraints:
            return {
                "error": f"json_constraints not found for {operator_name}",
                "cases_path": None,
                "cases_count": None,
            }

        context = parse_result_json(constraints)
        cases = TestCaseGenerator(context, seed=seed).generate(count=count)
        cases_json = json.dumps(
            [c.model_dump() for c in cases], ensure_ascii=False,
        )
        save_result = await _mcp_client.save_test_cases(
            operator_name=operator_name,
            cases_json=cases_json,
            source="generated",
        )
        out_path = save_result.get("output_path", "")
        logger.info(
            "generate_cases: %s → %d cases at %s",
            operator_name, len(cases), out_path,
        )
        return {
            "error": None,
            "cases_path": out_path,
            "cases_count": len(cases),
        }

    except Exception as e:
        logger.exception("generate_cases failed for %s", operator_name)
        return {
            "error": str(e),
            "cases_path": None,
            "cases_count": None,
        }
