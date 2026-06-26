"""ConstraintCheckAgent: post-pipeline consistency checker.

Uses DeepAgent + LocalShellBackend (same pattern as op-batch project).
The agent reads JSON/Markdown files itself and writes HTML report to disk.
This avoids putting ~160KB of content into a single LLM call.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from agent.core.llm import create_constraint_check_llm

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)

_constraint_check_agent: Any = None

# Global lock: ensures only one constraint check runs at a time.
# Prevents concurrent checks from automatic (task_engine) + manual (API) triggers.
_check_lock = asyncio.Lock()

# Temp directory for writing Markdown/JSON files for the agent to read
_TEMP_DIR = os.path.join(_PROJECT_ROOT, "data", "check_temp")


# ---------------------------------------------------------------------------
# System prompt (short — checking rules are in the user prompt, not system)
# ---------------------------------------------------------------------------

_CHECKER_SYSTEM_PROMPT = (
    "You are a professional operator constraint review assistant. "
    "You analyze operator documentation and generate HTML checklist reports."
)


def _get_constraint_check_agent() -> Any:
    """Lazily create and cache the DeepAgent with LocalShellBackend."""
    global _constraint_check_agent
    if _constraint_check_agent is not None:
        return _constraint_check_agent

    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend

    backend = LocalShellBackend(
        root_dir=_PROJECT_ROOT,
        virtual_mode=False,
        inherit_env=True,
        timeout=300,
    )

    _constraint_check_agent = create_deep_agent(
        model=create_constraint_check_llm(),
        backend=backend,
        system_prompt=_CHECKER_SYSTEM_PROMPT,
    )
    logger.info("ConstraintCheckAgent: created (with LocalShellBackend)")
    return _constraint_check_agent


# ---------------------------------------------------------------------------
# Prompt template (checking rules in user message, like op-batch)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """Your task is to analyze the constraint extraction results for operator **{operator_name}** and generate an HTML checklist report.

## Context

- JSON data file (extraction result): {json_path}
- Original Markdown doc (ground truth): {md_path}
- Output file: {output_path}

## Workflow

1. Read the JSON data file, focus on the `constraints_in_parameters` field and `inputs`/`outputs` fields.
2. Read the original Markdown document, focus on:
   - The parameter table in the GetWorkspaceSize section: each parameter's **使用说明** column contains constraint descriptions
   - The constraints section (if present)
   - The function prototype section: function signatures define actual parameters
3. Perform the following two-dimension analysis
4. Write the analysis result as HTML to: {output_path}

## Analysis Dimension 1: Missing Constraints

Systematically extract ALL constraint relationships from the original Markdown document, then check whether each one is represented in `constraints_in_parameters`. For each parameter's **使用说明** column, look for these constraint patterns:
- Cross-parameter: "与...一致", "与...相同", "对应关系"
- Intra-list consistency: "该参数中所有Tensor的...保持一致"
- Shape size vs dimension count: "shape size" = element count, NOT len(shape)
- dtype correspondence: conditional dtype relationships
- Element count: "元素个数为1"
- Value range: "取值范围", "大于", "小于"

For each missing constraint, provide: parameter name, original text, suggested expression, why it matters.

## Analysis Dimension 2: Expression Consistency

For EACH constraint in `constraints_in_parameters`, verify:
- Semantic accuracy: does the expression match the original text meaning?
- expr_type correctness: type_equality for dtype, shape_dependency for shape, etc.
- Parameter reference validity: all referenced params must exist
- src_text accuracy: does it match the original document

## Output Format

Generate a bright-style HTML report with:
1. Header with operator name
2. Navigation bar (sticky)
3. Summary statistics (pass/warn/fail counts)
4. Dimension 1: Missing constraints table
5. Dimension 2: Expression consistency table (one row per constraint)
6. Conclusion

All CSS must be inline. No external resources. Use green=pass, orange=warn, red=fail color coding.

Write the complete HTML to: {output_path}
"""


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

async def run_constraint_check(
    markdown_content: str,
    json_constraints: str,
    operator_name: str,
) -> str:
    """Run consistency check, return HTML report string.

    Uses a global asyncio.Lock to ensure only one check runs at a time.
    This prevents concurrent checks from automatic (task_engine) and
    manual (API) triggers overwhelming the LLM API.
    """
    if not markdown_content.strip() or not json_constraints.strip():
        logger.warning("ConstraintCheckAgent: empty input for %s", operator_name)
        return ""

    async with _check_lock:
        return await _run_constraint_check_impl(markdown_content, json_constraints, operator_name)


async def _run_constraint_check_impl(
    markdown_content: str,
    json_constraints: str,
    operator_name: str,
) -> str:
    """Actual check implementation (called under _check_lock)."""
    if not markdown_content.strip() or not json_constraints.strip():
        logger.warning("ConstraintCheckAgent: empty input for %s", operator_name)
        return ""

    # Ensure temp dir exists
    os.makedirs(_TEMP_DIR, exist_ok=True)

    # Write content to temp files so the agent can read them
    md_path = os.path.join(_TEMP_DIR, f"{operator_name}.md")
    json_path = os.path.join(_TEMP_DIR, f"{operator_name}.json")
    html_path = os.path.join(_TEMP_DIR, f"{operator_name}_report.html")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_constraints)

    # Use relative paths from project root for the agent
    rel_md = os.path.relpath(md_path, _PROJECT_ROOT)
    rel_json = os.path.relpath(json_path, _PROJECT_ROOT)
    rel_html = os.path.relpath(html_path, _PROJECT_ROOT)

    prompt = _PROMPT_TEMPLATE.format(
        operator_name=operator_name,
        json_path=rel_json,
        md_path=rel_md,
        output_path=rel_html,
    )

    agent = _get_constraint_check_agent()

    logger.info(
        "ConstraintCheckAgent: starting check for %s (md=%d chars, json=%d chars, temp=%s)",
        operator_name, len(markdown_content), len(json_constraints), _TEMP_DIR,
    )

    try:
        from langchain_core.messages import HumanMessage
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": 500},
        )
        logger.info("ConstraintCheckAgent: agent returned for %s", operator_name)

        # Read the HTML file the agent wrote
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            logger.info(
                "ConstraintCheckAgent: check completed for %s (%d chars HTML)",
                operator_name, len(html),
            )
            return html
        else:
            # Agent didn't write the file, try extracting from last message
            logger.warning(
                "ConstraintCheckAgent: no HTML file at %s, extracting from message",
                html_path,
            )
            msgs = result.get("messages", [])
            for m in reversed(msgs):
                if hasattr(m, "content") and m.content and "<html" in str(m.content).lower():
                    return str(m.content)
            return ""

    except Exception as e:
        logger.exception("ConstraintCheckAgent: failed for %s: %s", operator_name, e)
        return ""
