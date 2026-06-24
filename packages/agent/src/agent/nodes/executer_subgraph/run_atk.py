"""Step 3 of ExecuterAgent: execute ATK test cases on remote machine.

Currently a mock implementation with logging. Will be replaced with actual
remote execution logic in the future.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)


async def exec_run_atk_node(state: PipelineState) -> dict[str, Any]:
    """Execute ATK test cases on execution machine (mock)."""
    if state.get("error"):
        return {"error": state.get("error")}

    operator_name = state.get("operator_name", "")
    executor_path = state.get("atk_executor_path")
    cases_count = state.get("cases_count", 0)

    if not executor_path:
        return {"error": "atk_executor_path is required"}

    logger.info("exec_run_atk: executing ATK for %s (%d cases)", operator_name, cases_count)

    try:
        logger.info("exec_run_atk: [MOCK] connecting to execution machine...")
        await asyncio.sleep(0.5)

        logger.info("exec_run_atk: [MOCK] uploading %s", executor_path)
        await asyncio.sleep(0.3)

        logger.info("exec_run_atk: [MOCK] running atk --executor %s", executor_path)
        await asyncio.sleep(1.0)

        passed = cases_count
        failed = 0
        total = cases_count

        logger.info(
            "exec_run_atk: [MOCK] execution complete — %d/%d passed",
            passed, total,
        )

        return {
            "exec_result": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "status": "mock_success",
            },
            "error": None,
        }
    except Exception as e:
        logger.exception("exec_run_atk failed for %s", operator_name)
        return {"error": str(e)}
