"""ComplexRelationAgent: DeepAgent for complex parameter relation expressions.

Follows the exact same pattern as dimensions_agent.py:
- Knowledge base (skill.md files) is eager-loaded into the system prompt
- tools=[] -- Agent only generates expr; Phase 0 validation in Python
- Lazy singleton agent instance
- Adding a new pattern = dropping a new .md file, no code change

Handles:
- Type 3: self-constraints (scalar ranges, enums, dim limits)
- Type 4: complex conditional relations (enum + presence + shape)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from agent.utils.llm_common import JSON_BLOCK_RE

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
)

_complex_relation_agent: Any = None


def _load_relation_skills() -> str:
    """Eager-load all .md files under knowledge/relation_skills/."""
    kb_dir = os.path.join(_PROJECT_ROOT, "knowledge", "relation_skills")
    if not os.path.isdir(kb_dir):
        logger.warning("ComplexRelationAgent: knowledge dir not found: %s", kb_dir)
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
                logger.warning("ComplexRelationAgent: cannot read %s", fpath)
    return "\n\n---\n\n".join(parts)


_COMPLEX_RELATION_SYSTEM_PROMPT = (
    "You are a complex parameter relation expression builder for CANN operators.\n"
    "\n"
    "Your task: generate a Python boolean expression (expr) for parameter\n"
    "relations that single-shot LLM cannot reliably handle.\n"
    "\n"
    "## Context\n"
    "You will receive:\n"
    "- relation: description, source_citation, relation_type, params\n"
    "- implicit_params: named dimension variables and their mappings\n"
    "- param_shapes: shape info for each parameter\n"
    "- signatures: function signature text\n"
    "\n"
    "## Output Format (MUST follow exactly)\n"
    "Return a JSON object:\n"
    '{"expr_type": "...", "expr": "...", "confidence": "high/medium/low"}\n'
    "\n"
    "Apply the rules and examples from the Knowledge Base below.\n"
)


def _get_complex_relation_agent() -> Any:
    """Lazily create and cache the DeepAgent.

    tools=[] -- Agent only generates expr; Phase 0 validation happens
    in Python after the Agent returns (same pattern as dimensions_agent).
    """
    global _complex_relation_agent
    if _complex_relation_agent is not None:
        return _complex_relation_agent

    from deepagents import create_deep_agent
    from agent.core.llm import create_llm

    kb = _load_relation_skills()
    system_prompt = _COMPLEX_RELATION_SYSTEM_PROMPT
    if kb:
        system_prompt = system_prompt + "\n\n## Knowledge Base\n\n" + kb

    _complex_relation_agent = create_deep_agent(
        model=create_llm(),
        tools=[],
        system_prompt=system_prompt,
        name="complex-relation-agent",
    )
    logger.info("ComplexRelationAgent: created (KB=%d chars)", len(kb))
    return _complex_relation_agent


def _parse_agent_response(text: str) -> dict[str, str]:
    """Parse agent output into {expr_type, expr, confidence} dict."""
    text = text.strip()

    match = JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "expr_type": data.get("expr_type", ""),
                "expr": data.get("expr", ""),
                "confidence": data.get("confidence", "high"),
                "uncertainty_reason": data.get("uncertainty_reason", ""),
            }
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict):
                return {
                    "expr_type": data.get("expr_type", ""),
                    "expr": data.get("expr", ""),
                    "confidence": data.get("confidence", "high"),
                    "uncertainty_reason": data.get("uncertainty_reason", ""),
                }
        except json.JSONDecodeError:
            pass

    logger.warning("ComplexRelationAgent: failed to parse response: %s", text[:200])
    return {"expr_type": "", "expr": "", "confidence": "low"}


async def _extract_ai_text(result: dict) -> str:
    """Extract the last AI message content from agent result."""
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        if hasattr(m, "content") and m.content:
            return m.content
    return ""


async def generate_expr_via_agent(
    rel: dict,
    signatures_text: str,
    param_shapes_text: str,
    implicit_params_text: str,
) -> dict[str, str]:
    """Generate expr for a complex relation (type 4) via DeepAgent.

    Returns {"expr_type", "expr", "confidence"} dict.
    Falls back to empty expr on any failure.
    """
    user_msg = (
        "## Relation\n"
        "- relation_type: " + rel.get("relation_type", "") + "\n"
        "- params: " + json.dumps(rel.get("params", []), ensure_ascii=False) + "\n"
        "- description: " + rel.get("description", "") + "\n"
        "- source_citation: " + rel.get("source_citation", "") + "\n\n"
        "## Function Signatures\n" + signatures_text + "\n\n"
        "## Parameter Shapes\n" + param_shapes_text + "\n\n"
        + implicit_params_text + "\n"
    )

    try:
        agent = _get_complex_relation_agent()
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_msg}],
        })
        ai_text = await _extract_ai_text(result)
        return _parse_agent_response(ai_text)
    except Exception:
        logger.exception("ComplexRelationAgent: invocation failed")
        return {"expr_type": "", "expr": "", "confidence": "low"}


async def generate_self_constraints_via_agent(
    params: list[dict],
    signatures_text: str,
    param_shapes_text: str,
    implicit_params_text: str,
) -> list[dict]:
    """Generate self-constraint exprs (type 3) via DeepAgent.

    Batch-processes uncovered parameters in one Agent call.
    Returns a list of relation dicts with relation_object populated.
    """
    if not params:
        return []

    param_lines = []
    for i, p in enumerate(params, 1):
        desc = p.get("param_desc", "") or p.get("llm_description", "")
        param_lines.append(
            str(i) + ". param_name=" + p.get("param_name", "")
            + ", param_type=" + p.get("param_type", "")
            + ", description=" + desc
        )

    user_msg = (
        "## Task\n"
        "Extract self-constraints for the following parameters.\n"
        "Each parameter may have zero or more constraints.\n"
        "Return a JSON array of constraint objects.\n\n"
        "## Parameters\n"
        + "\n".join(param_lines) + "\n\n"
        "## Function Signatures\n" + signatures_text + "\n\n"
        "## Parameter Shapes\n" + param_shapes_text + "\n\n"
        + implicit_params_text + "\n"
    )

    try:
        agent = _get_complex_relation_agent()
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_msg}],
        })
        ai_text = await _extract_ai_text(result)
    except Exception:
        logger.exception("ComplexRelationAgent: self-constraint invocation failed")
        return []

    text = ai_text.strip()
    match = JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        arr_match = re.search(r"\[[\s\S]*\]", text)
        if arr_match:
            try:
                data = json.loads(arr_match.group(0))
            except json.JSONDecodeError:
                logger.warning(
                    "ComplexRelationAgent: failed to parse self-constraint response: %s",
                    text[:200],
                )
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    results: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        expr_type = item.get("expr_type", "")
        expr = item.get("expr", "")
        desc = item.get("description", "")
        src = item.get("source_citation", "")
        pname = item.get("param_name", "")
        if not expr_type and not desc:
            continue
        params_list = [pname] if pname else item.get("params", [])
        results.append({
            "function_name": "",
            "relation_type": "self_constraint",
            "platform": item.get("platform", ""),
            "description": desc,
            "params": params_list,
            "param_optional": {pname: False} if pname else {},
            "source_citation": src,
            "relation_object": {
                "expr_type": expr_type,
                "expr": expr,
                "relation_params": params_list,
                "src_text": src,
            },
            "_source": "agent_self_constraint",
        })

    logger.info(
        "ComplexRelationAgent: generated %d self-constraints from %d params",
        len(results), len(params),
    )
    return results
