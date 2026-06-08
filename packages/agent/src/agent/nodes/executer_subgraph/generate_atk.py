"""Step 1 of ExecuterAgent: generate ATK executor script.

Runs ``python generator.py <cases.json> --signatures aclnn_extracted.txt``
to produce ``{operator_name}_atk_executor.py``.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
_GENERATOR_SCRIPT = _RESOURCES_DIR / "generator.py"
_SIGNATURES_FILE = _RESOURCES_DIR / "aclnn_extracted.txt"
_OUTPUT_DIR = Path(__file__).resolve().parents[5] / "executors"


def _run_generator(cmd: list[str]) -> tuple[int, str, str]:
    """Run generator.py subprocess (blocking, for use in thread executor)."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout, proc.stderr


async def exec_generate_atk_node(state: PipelineState) -> dict[str, Any]:
    """Generate ATK executor .py file from cases JSON + signature table."""
    if state.get("error"):
        return {"error": state.get("error")}

    operator_name = state.get("operator_name", "")
    cases_path = state.get("cases_path")
    if not operator_name:
        return {"error": "operator_name is required"}
    if not cases_path:
        return {"error": "cases_path is required — run GeneratorAgent first"}

    logger.info("exec_generate_atk: generating ATK executor for %s", operator_name)

    try:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = _OUTPUT_DIR / f"{operator_name}_atk_executor.py"

        cmd = [
            sys.executable,
            str(_GENERATOR_SCRIPT),
            cases_path,
            "-o", str(output_file),
            "--signatures", str(_SIGNATURES_FILE),
        ]

        returncode, stdout, stderr = await asyncio.to_thread(_run_generator, cmd)

        if returncode != 0:
            logger.error("exec_generate_atk: generator.py failed: %s", stderr)
            return {"error": f"generator.py failed: {stderr}"}

        logger.info(
            "exec_generate_atk: generated %s", output_file,
        )
        return {
            "atk_executor_path": str(output_file),
            "atk_executor_code": output_file.read_text(encoding="utf-8"),
            "error": None,
        }
    except Exception as e:
        logger.exception("exec_generate_atk failed for %s", operator_name)
        return {"error": str(e)}
