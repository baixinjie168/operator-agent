"""Step 5 of GeneratorAgent: generate the test case data.

Wraps the existing ``TestCaseGenerator``: parses the constraints, runs the
sampler to produce ``count`` cases, and persists them to the DB + disk via MCP.
Cases are saved per-product: ``cases/{op}_cases_{product}.json``
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agent.generators import TestCaseGenerator, parse_result_json
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_DEFAULT_COUNT = 10
_DEFAULT_SEED = 42


def _sanitize_product_name(product: str) -> str:
    """Convert product name to a safe filename component."""
    # Replace slashes and special characters with underscores
    safe = re.sub(r'[/\\:*?"<>|]', '_', product)
    # Remove extra whitespace
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe or "default"


async def case_generate_node(state: PipelineState) -> dict[str, Any]:
    """Run the TestCaseGenerator and persist results to MCP + disk."""
    if state.get("error"):
        return {"error": state.get("error")}

    operator_name = state.get("operator_name", "")
    constraints = state.get("constraints_raw")
    if not operator_name or not constraints:
        return {"error": "operator_name or constraints_raw missing"}

    count = int(state.get("cases_count") or state.get("count") or _DEFAULT_COUNT)
    seed = int(state.get("cases_seed") or state.get("seed") or _DEFAULT_SEED)

    logger.info(
        "case_generate: running TestCaseGenerator for %s (count=%d, seed=%d)",
        operator_name, count, seed,
    )

    try:
        context = parse_result_json(constraints)
        logger.info(
            "case_generate: operator=%s, requested count=%d, platforms=%d",
            operator_name, count, len(context.supported_platforms) if context.supported_platforms else 1,
        )
        cases = TestCaseGenerator(context, seed=seed).generate(count=count)

        # Group cases by supported_product
        cases_by_product: dict[str, list] = {}
        for c in cases:
            case_data = c.model_dump()
            product = case_data.get("supported_product", "") or "default"
            if product not in cases_by_product:
                cases_by_product[product] = []
            cases_by_product[product].append(case_data)

        # Save per-product files
        output_paths = []
        for product, product_cases in cases_by_product.items():
            safe_product = _sanitize_product_name(product)
            cases_json = json.dumps(product_cases, ensure_ascii=False)

            # Save via MCP with product-specific filename
            save_result = await _mcp_client.save_test_cases(
                operator_name=f"{operator_name}_{safe_product}",
                cases_json=cases_json,
                source="generated",
            )
            out_path = save_result.get("output_path", "")
            output_paths.append(out_path)
            logger.info(
                "case_generate: %s [%s] → %d cases at %s",
                operator_name, product, len(product_cases), out_path,
            )

        # Use the first product's path as the main cases_path for backward compatibility
        out_path = output_paths[0] if output_paths else ""

        # Save to database immediately (don't wait for route to do it)
        try:
            from agent.db import save_test_cases as db_save_test_cases
            # Use the current task's run_id from state
            task_id = state.get("run_id")
            if task_id:
                db_save_test_cases(
                    task_id=task_id,
                    operator_name=operator_name,
                    cases=[c.model_dump() for c in cases],
                    constraint_doc_id=state.get("doc_id"),
                )
                logger.info("Saved %d test cases to DB for task %s", len(cases), task_id)
            else:
                logger.warning("No run_id in state, skipping DB save in node")
        except Exception as db_err:
            logger.warning("Failed to save cases to DB in node: %s", db_err)

        logger.info(
            "case_generate: %s → %d total cases across %d products",
            operator_name, len(cases), len(cases_by_product),
        )
        return {
            "cases": [c.model_dump() for c in cases],
            "cases_path": out_path,
            "cases_count": len(cases),
            "error": None,
        }
    except Exception as e:
        logger.exception("case_generate failed for %s", operator_name)
        return {"error": str(e), "cases_path": None, "cases_count": None}
