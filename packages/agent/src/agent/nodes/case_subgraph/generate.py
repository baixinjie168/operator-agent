"""Step 5 of GeneratorAgent: generate the test case data.

Wraps the existing ``TestCaseGenerator``: parses the constraints, runs the
sampler to produce ``count`` cases, and persists them to the DB + disk via MCP.
Cases are saved per-product: ``cases/{op}_cases_{product}.json``
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.core.config import settings
from agent.generators import TestCaseGenerator
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
    # Optional narrow filter: when set, only generate for this product and skip
    # the others. Used by ``scripts/batch_verify`` so ``--count`` reflects the
    # target product's execution count instead of ``count * num_products``.
    target_product = state.get("target_product") or None

    logger.info(
        "case_generate: running TestCaseGenerator for %s (count=%d, seed=%d, target_product=%s)",
        operator_name, count, seed, target_product or "ALL",
    )

    try:
        # 直接透传原始 ``json_constraints`` dict，不再做 ``parse_result_json``
        # 或任何 ``GeneratorContext`` 中间层转换；按平台分组的用例由 facade
        # 的 ``generate_by_platform`` 直接给出。
        gen = TestCaseGenerator(constraints, seed=seed)
        logger.info(
            "case_generate: operator=%s, requested count=%d, platforms=%d, target=%s",
            operator_name, count, len(gen.supported_platforms) or 1,
            target_product or "ALL",
        )
        jsonl_save_path = str(settings.cases_dir / operator_name)
        if target_product:
            # Narrow generation to a single product so the result count is
            # exactly ``count`` instead of ``count * num_products``.
            cases_by_product = {
                target_product: gen.generate_for_platform(
                    target_product, count, jsonl_save_path=jsonl_save_path,
                ),
            }
        else:
            cases_by_product = gen.generate_by_platform(
                count=count, jsonl_save_path=jsonl_save_path,
            )

        # Save per-product files. 注入 ``supported_product`` 字段到每条用例 dict，
        # 这样 ``db.save_test_cases`` / ``/api/v1/test-cases?supported_product=...`` 能按
        # 产品过滤，前端弹框也可以按产品下拉切换展示。
        output_paths = []
        all_case_dicts: list[dict[str, Any]] = []
        for product, product_cases in cases_by_product.items():
            safe_product = _sanitize_product_name(product)
            product_case_dicts = [c.model_dump() for c in product_cases]
            for case_dict in product_case_dicts:
                # 始终以产品名为准；若 CaseConfig 自带 supported_product 也以平台循环变量覆盖，
                # 避免 generator 输出遗漏或错乱。
                case_dict["supported_product"] = product
            all_case_dicts.extend(product_case_dicts)
            cases_json = json.dumps(product_case_dicts, ensure_ascii=False)

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
        # 按产品用例分布日志（便于核对 "x个产品共生成y个用例" 文案）
        per_product_count = {p: len(cs) for p, cs in cases_by_product.items()}
        logger.info(
            "case_generate summary: operator=%s products=%s",
            operator_name, per_product_count,
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
                    cases=all_case_dicts,
                    constraint_doc_id=state.get("doc_id"),
                )
                logger.info("Saved %d test cases to DB for task %s", len(all_case_dicts), task_id)
            else:
                logger.warning("No run_id in state, skipping DB save in node")
        except Exception as db_err:
            logger.warning("Failed to save cases to DB in node: %s", db_err)

        logger.info(
            "case_generate: %s → %d total cases across %d products",
            operator_name, len(all_case_dicts), len(cases_by_product),
        )
        return {
            "cases": all_case_dicts,
            "cases_path": out_path,
            "cases_count": len(all_case_dicts),
            "error": None,
        }
    except Exception as e:
        logger.exception("case_generate failed for %s", operator_name)
        return {"error": str(e), "cases_path": None, "cases_count": None}
