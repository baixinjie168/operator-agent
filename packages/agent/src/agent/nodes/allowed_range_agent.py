"""AllowedRangeAgent: DeepAgent-based batch extraction of allowed_range_value.

Replaces the per-param LLM calls in allowed_range_extract_node with a
batch Agent approach that:
1. Eager-loads knowledge/allowed_range/*.md into the system prompt
2. Processes all params in a function group in one Agent call
3. Falls back to per-param extraction on alignment mismatch
4. Validates structure via Python (_validate_range_structure)

Pattern follows dimensions_agent.py (lazy-create + cache DeepAgent).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agent.utils.llm_common import create_llm, parse_json_response

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)

_allowed_range_agent: Any = None


def _load_allowed_range_knowledge() -> str:
    """Read all .md files under knowledge/allowed_range/ at agent creation.

    The content is appended to the system prompt so the LLM sees every rule
    and example without needing a read_file tool-call round-trip.

    Adding a new special-case rule = dropping a new .md file into
    knowledge/allowed_range/examples/ -- no code change required.
    """
    kb_dir = os.path.join(_PROJECT_ROOT, "knowledge", "allowed_range")
    if not os.path.isdir(kb_dir):
        logger.warning("AllowedRangeAgent: knowledge dir not found: %s", kb_dir)
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
                logger.warning("AllowedRangeAgent: cannot read %s", fpath)
    return "\n\n---\n\n".join(parts)


_ALLOWED_RANGE_SYSTEM_PROMPT = """You are an allowed_range_value extraction expert for CANN operator documents.

For each parameter listed in the user message, extract its value range
from the provided document section text.

## Output Format (MUST follow exactly)

Return a JSON array. Each element:
{"param_name": "...", "platform": "...", "allowed_range_value": "...", "type": "range|enum"}

Rules:
- Only extract value range / value constraint / enum value constraints
- Ignore shape, dtype, format constraints
- For enum type: split separators (/ 、 以及 and) into comma-separated values
- For platform-specific values: set platform field to the platform name
- For no platform restriction: set platform to ""
- If a parameter has no value range info, do NOT include it in the output

Apply the rules and examples from the Knowledge Base below.
"""


def _get_allowed_range_agent() -> Any:
    """Lazily create and cache the DeepAgent for allowed_range extraction."""
    global _allowed_range_agent
    if _allowed_range_agent is not None:
        return _allowed_range_agent

    from deepagents import create_deep_agent

    kb = _load_allowed_range_knowledge()
    system_prompt = _ALLOWED_RANGE_SYSTEM_PROMPT
    if kb:
        system_prompt = system_prompt + "\n\n## Knowledge Base\n\n" + kb

    _allowed_range_agent = create_deep_agent(
        model=create_llm(),
        tools=[],
        system_prompt=system_prompt,
        name="allowed-range-agent",
    )
    logger.info("AllowedRangeAgent: created (KB=%d chars)", len(kb))
    return _allowed_range_agent


def _extract_ai_text(result: dict) -> str:
    """Extract the last AI message text from a DeepAgent result."""
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        if hasattr(m, "content") and m.content:
            return m.content
    return ""


def _parse_batch_response(
    ai_text: str,
    expected_params: list[dict],
) -> dict[str, list[dict]]:
    """Parse Agent batch response into per-param results.

    Returns: {param_name: [{platform, allowed_range_value, type}, ...]}
    """
    data = parse_json_response(ai_text, list)
    if not isinstance(data, list):
        logger.warning("AllowedRangeAgent: failed to parse batch response: %s", ai_text[:200])
        return {}

    by_param: dict[str, list[dict]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        pname = item.get("param_name", "")
        if not pname:
            continue
        entry = {
            "platform": item.get("platform", ""),
            "allowed_range_value": item.get("allowed_range_value", ""),
            "type": item.get("type", "range"),
        }
        by_param.setdefault(pname, []).append(entry)

    expected_names = {p.get("param_name", "") for p in expected_params}
    found_names = set(by_param.keys())
    missing = expected_names - found_names
    if missing:
        logger.info(
            "AllowedRangeAgent: batch missing %d params: %s",
            len(missing), ", ".join(sorted(missing)),
        )

    return by_param


async def _extract_batch_via_agent(
    params: list[dict],
    sections_text: str,
) -> dict[str, list[dict]]:
    """Batch-extract all params in one Agent call.

    Returns: {param_name: [{platform, allowed_range_value, type}, ...]}
    """
    if not params or not sections_text.strip():
        return {}

    param_list = json.dumps(
        [{"name": p.get("param_name", ""), "type": p.get("param_type", "")} for p in params],
        ensure_ascii=False,
    )
    user_msg = (
        f"Extract allowed_range_value for each parameter below "
        f"from the document section text.\n\n"
        f"Parameters:\n{param_list}\n\n"
        f"Document section text:\n{sections_text}"
    )

    try:
        agent = _get_allowed_range_agent()
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_msg}],
        })
        ai_text = _extract_ai_text(result)
        parsed = _parse_batch_response(ai_text, params)

        expected_names = {p.get("param_name", "") for p in params}
        missing = expected_names - set(parsed.keys())
        if missing:
            logger.info(
                "AllowedRangeAgent: batch missed %d/%d params, falling back to per-param",
                len(missing), len(expected_names),
            )
            for p in params:
                pname = p.get("param_name", "")
                if pname in missing:
                    single = await _extract_one_via_agent(p, sections_text)
                    if single:
                        parsed[pname] = single

        return parsed

    except Exception as e:
        logger.exception("AllowedRangeAgent: batch invocation failed: %s", e)
        result_map: dict[str, list[dict]] = {}
        for p in params:
            pname = p.get("param_name", "")
            single = await _extract_one_via_agent(p, sections_text)
            if single:
                result_map[pname] = single
        return result_map


async def _extract_one_via_agent(
    param: dict,
    sections_text: str,
) -> list[dict]:
    """Fallback: extract a single param (isolated failure).

    Returns: [{platform, allowed_range_value, type}, ...]
    """
    pname = param.get("param_name", "")
    ptype = param.get("param_type", "")
    if not sections_text.strip():
        return []

    user_msg = (
        f"Extract allowed_range_value for this single parameter "
        f"from the document section text.\n\n"
        f"Parameter: {pname} (type: {ptype})\n\n"
        f"Document section text:\n{sections_text}"
    )

    try:
        agent = _get_allowed_range_agent()
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_msg}],
        })
        ai_text = _extract_ai_text(result)
        data = parse_json_response(ai_text, list)
        if not isinstance(data, list):
            logger.warning("AllowedRangeAgent: per-param parse failed for %s", pname)
            return []

        entries: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            entries.append({
                "platform": item.get("platform", ""),
                "allowed_range_value": item.get("allowed_range_value", ""),
                "type": item.get("type", "range"),
            })
        return entries

    except Exception:
        logger.warning("AllowedRangeAgent: per-param extraction failed for %s", pname)
        return []
