"""Semantic value range inference rules loader.

Loads rules from ``config/semantic_value_rules.yaml`` and
provides utilities for:
- Matching parameter descriptions against keyword rules
- Generating allowed_range_value for scalar parameters (Layer 1)
- Generating expr templates for Tensor element-level constraints (Layer 2)
- Building prompt context for LLM-based extraction

Usage::

    from agent.utils.semantic_rules import (
        match_rules,
        get_allowed_range_for_scalar,
        get_expr_for_tensor,
        build_prompt_context,
    )
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from agent.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Path to the semantic rules YAML file. Configurable via ``settings`` so it
# can be overridden with the ``SEMANTIC_RULES_FILE`` env var. CWD-relative by
# default, consistent with ``task_config_file`` and other project paths
# (operators_dir, database_path, etc.).
_RULES_PATH = Path(settings.semantic_rules_file)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_rules() -> list[dict[str, Any]]:
    """Load semantic inference rules from YAML file.

    Returns:
        List of rule dicts. Empty list if file is missing or invalid.

    The result is cached after the first call.
    """
    if not _RULES_PATH.exists():
        logger.warning(
            "semantic_value_rules.yaml not found at %s, using empty rules",
            _RULES_PATH,
        )
        return []

    try:
        with open(_RULES_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        logger.exception("Failed to load semantic_value_rules.yaml")
        return []

    if not isinstance(data, dict):
        logger.error("semantic_value_rules.yaml: expected dict, got %s", type(data))
        return []

    rules = data.get("rules", [])
    if not isinstance(rules, list):
        logger.error("semantic_value_rules.yaml: 'rules' must be a list")
        return []

    logger.info(
        "Loaded %d semantic value rules from %s (version=%s)",
        len(rules),
        _RULES_PATH,
        data.get("version", "unknown"),
    )
    return rules


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_rules(
    description: str,
    param_name: str = "",
) -> list[dict[str, Any]]:
    """Match a parameter description against all semantic rules.

    Searches for keyword hits in both ``description`` and ``param_name``
    (case-insensitive). A rule matches if any of its keywords appears as
    a substring.

    Args:
        description: The parameter's natural language description
            (e.g. from ``param_desc`` or ``llm_description``).
        param_name: The parameter name (e.g. ``expertTokensOptional``).
            Used as a secondary match target.

    Returns:
        List of matched rule dicts, sorted by confidence (high first).
        Empty list if no rules match.
    """
    if not description and not param_name:
        return []

    rules = load_rules()
    if not rules:
        return []

    search_text = f"{description} {param_name}".lower()

    matched: list[dict[str, Any]] = []
    for rule in rules:
        keywords = rule.get("keywords", [])
        for kw in keywords:
            if kw.lower() in search_text:
                matched.append(rule)
                break  # One keyword hit is enough

    # Sort by confidence: high > medium > low
    _CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
    matched.sort(key=lambda r: _CONFIDENCE_ORDER.get(r.get("confidence", "low"), 2))

    return matched


def get_allowed_range_for_scalar(
    description: str,
    param_name: str = "",
) -> list | None:
    """Get allowed_range_value for a scalar parameter via semantic rules.

    This is a Layer 1 helper: returns the ``allowed_range`` from the
    highest-confidence matched rule, or ``None`` if no rule matches.

    Args:
        description: Parameter description text.
        param_name: Parameter name.

    Returns:
        The ``allowed_range`` list (e.g. ``[[1, null]]``, ``[[0, 1]]``),
        or ``None`` if no rule matches or the rule has no ``allowed_range``.
    """
    matched = match_rules(description, param_name)
    for rule in matched:
        target = rule.get("target", "")
        if target in ("layer_1", "both"):
            ar = rule.get("allowed_range")
            if ar:
                return ar
    return None


def get_expr_for_tensor(
    description: str,
    param_name: str,
) -> dict[str, str] | None:
    """Get expr template for a Tensor parameter's element-level constraint.

    This is a Layer 2 helper: returns the formatted ``expr`` and
    ``expr_type`` from the highest-confidence matched rule.

    Args:
        description: Parameter description text.
        param_name: Parameter name (used to fill ``{param}`` placeholder).

    Returns:
        Dict with ``expr_type`` and ``expr`` keys, or ``None`` if no
        rule matches.
    """
    if not param_name:
        return None

    matched = match_rules(description, param_name)
    for rule in matched:
        target = rule.get("target", "")
        if target in ("layer_2", "both"):
            template = rule.get("expr_template", "")
            if template:
                expr = template.format(param=param_name)
                return {
                    "expr_type": rule.get("expr_type", "self_value_range"),
                    "expr": expr,
                    "description": rule.get("description", ""),
                    "confidence": rule.get("confidence", "low"),
                }
    return None


# ---------------------------------------------------------------------------
# Prompt context builder
# ---------------------------------------------------------------------------


def build_prompt_context() -> str:
    """Build a text block describing all semantic rules for LLM prompts.

    The output is intended to be injected into LLM prompts (e.g.
    ``ALLOWED_RANGE_VALUE_BUILD_PROMPT`` or ``SINGLE_PARAM_EXTRACT_PROMPT``)
    to give the LLM awareness of common semantic patterns.

    Returns:
        Multi-line string with rule descriptions, or empty string if
        no rules are loaded.
    """
    rules = load_rules()
    if not rules:
        return ""

    lines: list[str] = []
    lines.append("## 语义推断参考规则")
    lines.append("以下是常见的参数描述关键词与取值范围的对应关系，供提取时参考：")
    lines.append("")

    for rule in rules:
        target = rule.get("target", "none")
        if target == "none":
            continue  # Skip no-constraint rules

        keywords = ", ".join(rule.get("keywords", [])[:4])
        ar = rule.get("allowed_range", [])
        desc = rule.get("description", "")
        confidence = rule.get("confidence", "low")

        ar_str = str(ar) if ar else "N/A"
        lines.append(
            f"- [{confidence}] 关键词: {keywords} → allowed_range_value: {ar_str}"
        )
        if desc:
            lines.append(f"  说明: {desc}")

    return "\n".join(lines)
