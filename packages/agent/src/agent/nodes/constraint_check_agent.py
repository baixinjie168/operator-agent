"""ConstraintCheckAgent: post-pipeline consistency checker.

Uses a separate LLM model (constraint_check_llm_provider) to verify that
the generated JSON constraints are consistent with the original operator
Markdown document.  Produces an HTML report.

Pattern follows dimensions_agent.py (lazy-create + cache DeepAgent,
eager-load knowledge base into system prompt).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agent.core.llm import create_constraint_check_llm
from agent.utils.llm_common import parse_json_response

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)

_constraint_check_agent: Any = None


# ---------------------------------------------------------------------------
# Knowledge base eager-loader
# ---------------------------------------------------------------------------

def _load_checker_knowledge() -> str:
    """Read all .md files under knowledge/operator-constraint-checker/.

    The content is appended to the system prompt so the LLM sees every
    rule and example without needing a read_file tool-call round-trip.
    """
    kb_dir = os.path.join(_PROJECT_ROOT, "knowledge", "operator-constraint-checker")
    if not os.path.isdir(kb_dir):
        logger.warning("ConstraintCheckAgent: knowledge dir not found: %s", kb_dir)
        return ""
    parts: list[str] = []
    for root, _dirs, files in os.walk(kb_dir):
        for fname in sorted(files):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    parts.append(f.read().strip())
            except OSError:
                logger.warning("ConstraintCheckAgent: cannot read %s", fpath)
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt (concise - rules are in the knowledge base)
# ---------------------------------------------------------------------------

_CHECKER_SYSTEM_PROMPT = """You are a constraint checker for CANN operator documents.

## Important
In this context, the operator Markdown document and the JSON constraint
document are provided directly in the user message. Do NOT attempt to
read files. Work solely with the content provided.

## Task
Following the four-dimension checking rules in the Knowledge Base below,
perform a consistency check between the operator document and the JSON
constraints. Generate a bright-style HTML analysis report.

Return ONLY the complete HTML document. No other text.

## Knowledge Base
"""


def _get_constraint_check_agent() -> Any:
    """Lazily create and cache the DeepAgent for constraint checking.

    Uses a separate model (constraint_check_llm_provider) to avoid
    self-evaluation bias.
    """
    global _constraint_check_agent
    if _constraint_check_agent is not None:
        return _constraint_check_agent

    from deepagents import create_deep_agent

    kb = _load_checker_knowledge()
    system_prompt = _CHECKER_SYSTEM_PROMPT
    if kb:
        system_prompt = system_prompt + kb

    _constraint_check_agent = create_deep_agent(
        model=create_constraint_check_llm(),
        tools=[],
        system_prompt=system_prompt,
        name="constraint-checker",
    )
    logger.info(
        "ConstraintCheckAgent: created (KB=%d chars)",
        len(kb),
    )
    return _constraint_check_agent


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

def _extract_ai_text(result: dict) -> str:
    """Extract the last AI message text from a DeepAgent result."""
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        if hasattr(m, "content") and m.content:
            return m.content
    return ""


async def run_constraint_check(
    markdown_content: str,
    json_constraints: str,
    operator_name: str,
) -> str:
    """Run 4-dimension consistency check, return HTML report.

    Args:
        markdown_content: Full Markdown text of the operator document.
        json_constraints: JSON string of the generated constraints.
        operator_name: Operator name for context.

    Returns:
        HTML report string, or empty string on failure.
    """
    if not markdown_content.strip() or not json_constraints.strip():
        logger.warning("ConstraintCheckAgent: empty input for %s", operator_name)
        return ""

    agent = _get_constraint_check_agent()
    user_msg = (
        f"## 算子文档 (Markdown)\n{markdown_content}\n\n"
        f"## 约束文档 (JSON)\n{json_constraints}\n\n"
        "请按照检查规则，对以上算子文档和约束文档执行四维度一致性检查，"
        "生成明亮风格的HTML分析报告。只返回完整HTML，不要其他文字。"
    )

    try:
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_msg}],
        })
        html = _extract_ai_text(result)
        if html:
            logger.info("ConstraintCheckAgent: check completed for %s (%d chars HTML)",
                        operator_name, len(html))
        return html
    except Exception as e:
        logger.exception("ConstraintCheckAgent: failed for %s: %s", operator_name, e)
        return ""
