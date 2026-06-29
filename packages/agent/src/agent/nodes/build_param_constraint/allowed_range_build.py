"""AllowedRangeBuild node: extract allowed_range_value for all parameters.

Bool short-circuit + deterministic regex + YAML semantic rules + LLM extraction
with retry and validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from agent.core.config import settings
from agent.nodes.build_param_constraint._helpers import _normalize_type, _split_csv
from agent.nodes.build_param_constraint.state import BuildParamConstraintState
from agent.prompts import ALLOWED_RANGE_VALUE_BUILD_PROMPT
from agent.core.llm import create_llm
from agent.utils.llm_common import CONCURRENCY_LIMIT, parse_json_response
from agent.utils.param_validators import is_bool_type, is_tensor_type
from agent.utils.semantic_rules import (
    get_allowed_range_for_scalar,
    build_prompt_context,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bool narrowing helpers (Phase 0)
# ---------------------------------------------------------------------------

# Patterns that indicate a bool param cannot be True (or False).
# Matches: "暂不支持配为True", "不支持配为True", "不能为True", "仅支持False"
_BOOL_NOT_TRUE_RE = re.compile(
    r"不支持.*(?:配|设|置).*True|暂不支持.*True|不能.*True|仅支持.*False|只支持.*False",
    re.IGNORECASE,
)
_BOOL_NOT_FALSE_RE = re.compile(
    r"不支持.*(?:配|设|置).*False|暂不支持.*False|不能.*False|仅支持.*True|只支持.*True",
    re.IGNORECASE,
)


def _collect_bool_param_text(param: dict) -> str:
    """Collect all text that may contain bool constraint info for a param.

    usage_notes is stored as a JSON string like '{"*": "..."}' or
    '{"platform": "..."}'.  We parse and flatten it to plain text.
    """
    parts: list[str] = []
    for field in ("usage_notes", "llm_description", "param_desc"):
        raw = param.get(field, "") or ""
        if not raw:
            continue
        # usage_notes may be a JSON {platform: text} dict
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.startswith("{") else None
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            parts.extend(str(v) for v in parsed.values() if v)
        else:
            parts.append(str(raw))
    return " ".join(parts)


def _narrow_bool_from_text(text: str) -> bool | None:
    """Check if text constrains a bool param to a single value.

    Returns:
        False  — if text says "不支持配为True" (can only be False)
        True   — if text says "不支持配为False" (can only be True)
        None   — if no constraint found (keep default [True, False])
    """
    if not text:
        return None
    if _BOOL_NOT_TRUE_RE.search(text):
        return False
    if _BOOL_NOT_FALSE_RE.search(text):
        return True
    return None


# ---------------------------------------------------------------------------
# Phase 1: Deterministic preprocessing for allowed_range_value
# ---------------------------------------------------------------------------

# Pattern: (regex, result_or_lambda)
# For enum patterns, the lambda returns a tuple (value_list, "enum")
# For range patterns, the lambda returns just a list (implicit "range")
_RANGE_PATTERNS: list[tuple[str, Any]] = [
    # "枚举值: 1, 2, 3" → enum [[1,1], [2,2], [3,3]]
    (
        r"枚举值\s*[:：]\s*(.+)",
        lambda m: (
            [[int(v.strip()), int(v.strip())] for v in m.group(1).split(",") if v.strip().lstrip("-").isdigit()],
            "enum",
        ),
    ),
    # "只支持32/64" / "仅支持32、64" → enum [[32,32],[64,64]]
    # The "只支持/仅支持" keyword signals discrete values, not a range
    (
        r"(?:只支持|仅支持|只能|只能是)\s*(\d+(?:\s*[/、，,]\s*\d+)+)",
        lambda m: (
            [[int(v.strip()), int(v.strip())] for v in re.split(r"[/、，,]", m.group(1)) if v.strip().isdigit()],
            "enum",
        ),
    ),
    # "32/64" → enum [[32,32], [64,64]] (slash-separated discrete values)
    (
        r"(\d+)\s*/\s*(\d+)",
        lambda m: (
            [[int(m.group(1)), int(m.group(1))], [int(m.group(2)), int(m.group(2))]],
            "enum",
        ),
    ),
    # "范围0-100" / "0~100" / "[0, 100]" → range [[0, 100]]
    (
        r"\[?\s*(-?\d+)\s*[,，\-~]\s*(-?\d+)\s*\]?",
        lambda m: ([[int(m.group(1)), int(m.group(2))]], "range"),
    ),
]


def _try_deterministic_range(text: str) -> tuple[list, str] | None:
    """Try deterministic extraction of range from text.

    Returns:
        (value_list, ar_type) if pattern matches, None otherwise.
        ar_type is "range" or "enum".
    """
    for pattern, extractor in _RANGE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            try:
                result = extractor(m)
                # New format: (value_list, ar_type)
                if isinstance(result, tuple) and len(result) == 2:
                    value_list, ar_type = result
                    if value_list:
                        return value_list, ar_type
                # Legacy format: just a list (implicit "range")
                elif isinstance(result, list) and result:
                    return result, "range"
            except (ValueError, IndexError):
                continue
    return None


def _extract_param_sentences(text: str, param_name: str) -> str:
    """Extract sentences from *text* that mention *param_name*.

    Splits on Chinese/English sentence boundaries (。；;\\n) and
    returns only sentences containing the parameter name. This
    prevents the deterministic regex from matching another
    parameter's range (e.g. picking hWinSize's "7~32" when
    processing seqLength's "32/64").
    """
    if not text or not param_name:
        return ""
    # Also match camelCase → snake_case variants
    variants = {param_name}
    # Simple camel-to-snake for broader matching
    snake = re.sub(r"([A-Z])", r"_\1", param_name).lower().lstrip("_")
    if snake != param_name:
        variants.add(snake)

    sentences = re.split(r"[。；;。\n]", text)
    matched = [
        s.strip() for s in sentences
        if any(v in s for v in variants) and s.strip()
    ]
    return ". ".join(matched)


# ---------------------------------------------------------------------------
# Phase 1: Validation functions for allowed_range_value
# ---------------------------------------------------------------------------


def _validate_range_structure(
    ranges: list,
    param_type: str = "",
    ar_type: str = "range",
) -> tuple[bool, str]:
    """Validate structure of allowed_range_value array.

    Args:
        ranges: The value array to validate.
        param_type: C type of the parameter (for unsigned checks).
        ar_type: "range" (default) or "enum".
            - "range": each element is [min, max] with min <= max
            - "enum": each element is an exact array value (no min<=max check)

    Checks for "range":
    1. Each element is [min, max] with min <= max
    2. Values are int/float or null
    3. Type compatibility (unsigned types must be non-negative)
    4. Reasonable values (not exceeding 10^9)

    Checks for "enum":
    1. Each element is a list of int/float values
    2. Reasonable values (not exceeding 10^9)
    """
    if not isinstance(ranges, list):
        return False, "ranges must be a list"

    if not ranges:  # Empty is valid
        return True, ""

    for i, r in enumerate(ranges):
        # For enum type, None is a valid element (represents "empty configuration"
        # for aclIntArray params, e.g. "配置空或者[-2,-1]")
        if r is None:
            if ar_type == "enum":
                continue
            return False, f"range[{i}] must be a list, got NoneType"

        if not isinstance(r, list):
            return False, f"range[{i}] must be a list, got {type(r).__name__}"

        if ar_type == "enum":
            # Enum: each element is an exact array value
            for j, val in enumerate(r):
                if val is not None and not isinstance(val, (int, float)):
                    return False, f"range[{i}][{j}] must be int/float, got {type(val).__name__}"
                if val is not None and abs(val) > 1e9:
                    return False, f"range[{i}][{j}]: value {val} seems unreasonably large"
        else:
            # Range: each element is [min, max]
            if len(r) != 2:
                return False, f"range[{i}] must be [min, max], got {r}"

            min_val, max_val = r

            # Type check
            for j, val in enumerate([min_val, max_val]):
                if val is not None and not isinstance(val, (int, float)):
                    return False, f"range[{i}][{j}] must be int/float or null, got {type(val).__name__}"

            # min <= max check
            if min_val is not None and max_val is not None:
                if min_val > max_val:
                    return False, f"range[{i}]: min ({min_val}) > max ({max_val})"

            # Type compatibility check
            if "uint" in param_type.lower():
                for j, val in enumerate([min_val, max_val]):
                    if val is not None and val < 0:
                        return False, f"range[{i}][{j}]: negative value {val} for unsigned type"

            # Reasonable value check
            for j, val in enumerate([min_val, max_val]):
                if val is not None and abs(val) > 1e9:
                    return False, f"range[{i}][{j}]: value {val} seems unreasonably large"

    return True, ""


def _validate_range_source(
    ranges: list,
    context_text: str,
) -> tuple[bool, str]:
    """Validate that range values appear in source text.

    This is a soft check: empty ranges pass, missing values log warning but pass.
    """
    if not ranges:
        return True, ""  # Empty range is valid

    for r in ranges:
        min_val, max_val = r
        # Convert to string for text search
        for val in [min_val, max_val]:
            if val is not None:
                val_str = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
                if val_str not in context_text:
                    # Soft warning, don't fail
                    logger.debug(
                        "Range source validation: value %s not found in source text",
                        val_str,
                    )

    return True, ""


def _parse_allowed_range_response(text: str) -> dict[str, Any]:
    """Parse LLM response: {"type": "range"|"enum", "value": [[min,max], ...], "src_text": "..."}.

    Backward compatible: if LLM returns a plain array [[min,max], ...],
    treats it as {"type": "range", "value": [...]}.

    For "enum" type, null values are preserved (they represent "empty"
    configurations for aclIntArray params, e.g. "配置空或者[-2,-1]").
    For "range" type, non-list items are converted to [] (invalid ranges).
    """
    # Try parsing as JSON object with type+value
    data = parse_json_response(text, dict)
    if isinstance(data, dict):
        ar_type = data.get("type", "range")
        ar_value = data.get("value", [])
        src_text = data.get("src_text", "")
        if isinstance(ar_value, list):
            if ar_type == "enum":
                # Preserve None values; convert other non-list items to []
                ar_value = [
                    item if (isinstance(item, list) or item is None) else []
                    for item in ar_value
                ]
            else:
                ar_value = [item if isinstance(item, list) else [] for item in ar_value]
            return {"type": ar_type, "value": ar_value, "src_text": src_text}

    # Try parsing as plain array (backward compatible)
    data = parse_json_response(text, list)
    if isinstance(data, list):
        ar_value = [item if isinstance(item, list) else [] for item in data]
        return {"type": "range", "value": ar_value, "src_text": ""}

    logger.warning("BuildParamConstraint: failed to parse allowed_range: %s", text[:200])
    return {"type": "range", "value": [], "src_text": ""}


async def allowed_range_build_node(state: BuildParamConstraintState) -> dict[str, Any]:
    """Batch extract allowed_range_value for all params via LLM.

    Flow:
    1. Bool type params: short-circuit with [True, False]
    2. Deterministic preprocessing: regex-based extraction for common patterns
    3. LLM extraction with retry for remaining params
    4. Structural validation: min <= max, type compatibility
    5. Source validation: check if values appear in source text (soft check)

    Returns:
        Map of "fn::pn" → {"type": "range"|"enum", "value": [[min,max], ...]}.
    """
    params = state.get("params", [])
    constraints_text = state.get("constraints_text", "")
    if not params:
        return {"allowed_range_map": {}}

    _EMPTY: dict = {"type": "range", "value": []}
    result: dict[str, dict] = {}
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Phase 0: Separate bool params (short-circuit) and Tensor params (no value range)
    bool_params: list[dict] = []
    tensor_params: list[dict] = []
    remaining_params: list[dict] = []
    for p in params:
        ptype = p.get("param_type", "")
        if is_bool_type(ptype):
            bool_params.append(p)
        elif is_tensor_type(ptype):
            tensor_params.append(p)
        else:
            remaining_params.append(p)

    for p in bool_params:
        key = f"{p['function_name']}::{p['param_name']}"
        # Check usage_notes / llm_description for "不支持配为True/False" patterns
        # before defaulting to [True, False].
        bool_text = _collect_bool_param_text(p)
        narrowed = _narrow_bool_from_text(bool_text)
        if narrowed is not None:
            result[key] = {"type": "range", "value": [narrowed]}
        else:
            result[key] = {"type": "range", "value": [True, False]}

    # Tensor types have no scalar value range — dimensions describe shape rank,
    # not value bounds.  Skip all extraction phases to prevent the deterministic
    # regex from mis-matching dimension text (e.g. "2-8") as a value range.
    for p in tensor_params:
        key = f"{p['function_name']}::{p['param_name']}"
        result[key] = _EMPTY

    # Phase 0.5: Reuse allowed_range_value already extracted by node 4h
    # for string-type parameters (char*, const char*).  Node 4h's prompt
    # explicitly handles enumeration constraints ("枚举值限制"), while this
    # node's ALLOWED_RANGE_VALUE_BUILD_PROMPT only extracts numeric ranges.
    # Without this pass-through, string enum values are silently dropped.
    string_ar_count = 0
    still_remaining: list[dict] = []
    for p in remaining_params:
        ptype = _normalize_type(p.get("param_type", ""))
        ar_raw = p.get("allowed_range_value", "") or ""
        if ptype in ("char", "const char") and ar_raw.strip() and ar_raw.strip() != "[]":
            key = f"{p['function_name']}::{p['param_name']}"

            try:
                ar_list = json.loads(ar_raw) if isinstance(ar_raw, str) else ar_raw
            except (json.JSONDecodeError, TypeError):
                ar_list = []
            if ar_list:
                # Detect type from first item (default "range")
                ar_type = "range"
                for item in ar_list:
                    if isinstance(item, dict) and item.get("type") == "enum":
                        ar_type = "enum"
                        break

                # For char*/const char* params, "range" type values are almost
                # always string LENGTH constraints (e.g. "字符串长度要求(0, 128)"),
                # not value ranges.  A string parameter's legitimate value range
                # is always "enum" (discrete string values like activation names).
                # Skip "range" type to avoid putting length constraints into
                # allowed_range_value — they belong in constraints_in_parameters.
                if ar_type == "range":
                    result[key] = _EMPTY
                    string_ar_count += 1
                    logger.info(
                        "BuildParamConstraint: skipped range-type allowed_range for "
                        "char* param %s (likely string length constraint, not value range)",
                        p.get("param_name", ""),
                    )
                    continue

                if ar_type == "enum":
                    # Split comma-separated enum values into individual items
                    # and preserve platform dimension (Bug #1 + #2 fix).
                    # Each platform gets its own key: fn::pn::platform.
                    # Also store a generic-key fallback (first platform's values)
                    # in case platform names don't match exactly downstream.
                    first_split: list[str] = []
                    for item in ar_list:
                        if not isinstance(item, dict):
                            continue
                        plat = item.get("platform", "")
                        raw = item.get("allowed_range_value", "")
                        if not raw:
                            continue
                        split_values = _split_csv(raw)
                        if not split_values:
                            continue
                        if not first_split:
                            first_split = split_values
                        if plat:
                            result[f"{key}::{plat}"] = {"type": "enum", "value": split_values}
                        else:
                            result[key] = {"type": "enum", "value": split_values}
                    # Generic-key fallback for platform name mismatches
                    if first_split and key not in result:
                        result[key] = {"type": "enum", "value": first_split}
                else:
                    # Range type: keep raw text as-is (no splitting needed)
                    texts = [
                        item.get("allowed_range_value", "")
                        for item in ar_list
                        if isinstance(item, dict) and item.get("allowed_range_value")
                    ]
                    result[key] = {"type": "range", "value": texts if texts else []}
                string_ar_count += 1
                continue
        still_remaining.append(p)

    if string_ar_count > 0:
        logger.info(
            "BuildParamConstraint: reused %d string-type allowed_range values from node 4h",
            string_ar_count,
        )

    remaining_params = still_remaining

    # Phase 1: Deterministic preprocessing
    llm_needed: list[dict] = []
    deterministic_count = 0
    yaml_count = 0

    for p in remaining_params:
        llm_desc = p.get("llm_description", "") or ""
        if not llm_desc.strip() and not constraints_text.strip():
            key = f"{p['function_name']}::{p['param_name']}"
            result[key] = _EMPTY
            continue

        # aclIntArray params: skip deterministic regex — the regex matches
        # specific array values like [-2,-1] as numeric ranges, but for
        # aclIntArray these are enum array values, not continuous ranges.
        # Let the LLM handle them (prompt rule #9 covers this case).
        ptype = p.get("param_type", "")
        is_int_array = "aclIntArray" in ptype
        if is_int_array:
            llm_needed.append(p)
            continue

        deterministic = _try_deterministic_range(llm_desc)
        if deterministic is None:
            # Search only sentences mentioning this parameter, not the
            # full constraints text, to avoid matching another param's range
            param_name = p.get("param_name", "")
            param_sentences = _extract_param_sentences(constraints_text, param_name)
            deterministic = _try_deterministic_range(param_sentences)

        if deterministic is not None:
            det_value, det_type = deterministic
            key = f"{p['function_name']}::{p['param_name']}"
            param_type = p.get("param_type", "")
            is_valid, _ = _validate_range_structure(det_value, param_type, det_type)
            result[key] = {"type": det_type, "value": det_value if is_valid else []}
            deterministic_count += 1
        else:
            # Phase 1b: YAML semantic rules fallback for scalar params
            ptype = p.get("param_type", "")
            is_tensor = "aclTensor" in ptype
            if not is_tensor:
                # Combine llm_desc + param_desc for broader keyword coverage
                param_desc = p.get("param_desc", "") or ""
                yaml_search_text = f"{llm_desc}\n{param_desc}"
                yaml_ar = get_allowed_range_for_scalar(
                    yaml_search_text, p.get("param_name", "")
                )
                if yaml_ar:
                    key = f"{p['function_name']}::{p['param_name']}"
                    is_valid, _ = _validate_range_structure(yaml_ar, ptype)
                    result[key] = {"type": "range", "value": yaml_ar if is_valid else []}
                    yaml_count += 1
                    continue

            llm_needed.append(p)

    if deterministic_count > 0 or yaml_count > 0:
        logger.info(
            "BuildParamConstraint: deterministic range extraction handled %d/%d params "
            "(%d regex, %d yaml semantic rules)",
            deterministic_count + yaml_count, len(remaining_params),
            deterministic_count, yaml_count,
        )

    if not llm_needed:
        return {"allowed_range_map": result}

    # Phase 2: LLM extraction with retry + validation
    try:
        llm = create_llm()
    except Exception:
        logger.exception("BuildParamConstraint: failed to create LLM for allowed_range")
        for p in llm_needed:
            key = f"{p['function_name']}::{p['param_name']}"
            result[key] = _EMPTY
        return {"allowed_range_map": result}

    async def _extract_one_with_retry(param: dict) -> tuple[str, dict]:
        async with sem:
            key = f"{param['function_name']}::{param['param_name']}"
            c_type = param.get("param_type", "")
            llm_desc = param.get("llm_description", "") or ""

            context_parts: list[str] = []
            if constraints_text:
                context_parts.append(f"## 约束说明\n{constraints_text}")
            if llm_desc:
                context_parts.append(f"## 参数使用说明\n{llm_desc}")

            # Inject semantic rules context for LLM reference
            semantic_ctx = build_prompt_context()
            if semantic_ctx:
                context_parts.append(semantic_ctx)

            context_text = "\n\n".join(context_parts) if context_parts else ""
            if not context_text.strip():
                return key, _EMPTY

            # Retry loop
            for attempt in range(settings.dimensions_max_retries + 1):
                try:
                    prompt = ALLOWED_RANGE_VALUE_BUILD_PROMPT.format(
                        param_name=param["param_name"],
                        param_type=c_type,
                        context_text=context_text,
                    )
                    response = await llm.ainvoke(prompt)
                    text = response.content if hasattr(response, "content") else str(response)
                    parsed = _parse_allowed_range_response(text)
                    ar_type = parsed.get("type", "range")
                    ar_value = parsed.get("value", [])

                    # Structural validation
                    is_valid, error = _validate_range_structure(ar_value, c_type, ar_type)
                    if not is_valid:
                        if attempt < settings.dimensions_max_retries:
                            logger.warning(
                                "BuildParamConstraint: range validation failed for %s (attempt %d): %s",
                                param["param_name"], attempt + 1, error,
                            )
                            continue
                        else:
                            logger.warning(
                                "BuildParamConstraint: range validation failed after %d attempts for %s: %s",
                                settings.dimensions_max_retries + 1, param["param_name"], error,
                            )
                            return key, _EMPTY

                    # Source validation (soft check, only for range type)
                    if ar_type == "range":
                        _validate_range_source(ar_value, context_text)

                    return key, parsed

                except Exception:
                    if attempt < settings.dimensions_max_retries:
                        logger.warning(
                            "BuildParamConstraint: LLM call failed for %s (attempt %d)",
                            param["param_name"], attempt + 1,
                        )
                        continue
                    else:
                        logger.warning(
                            "BuildParamConstraint: LLM call failed after %d attempts for %s",
                            settings.dimensions_max_retries + 1, param["param_name"],
                        )
                        return key, _EMPTY

            return key, _EMPTY

    tasks = [_extract_one_with_retry(p) for p in llm_needed]
    llm_results = await asyncio.gather(*tasks)
    for key, value in llm_results:
        result[key] = value

    extracted = sum(1 for v in result.values() if v.get("value"))
    logger.info(
        "BuildParamConstraint: extracted allowed_range for %d/%d params "
        "(%d bool short-circuited, %d tensor skipped, %d deterministic, %d LLM)",
        extracted, len(params), len(bool_params), len(tensor_params),
        deterministic_count, len(llm_needed),
    )

    # NODE_PROGRESS: range_done — frontend ExtractorAgent panel
    from agent.runtime.context import get_context
    from agent.runtime.events import EventType, Span, SpanType
    ctx = get_context()
    if ctx and ctx.manager:
        span = Span(
            span_id="progress",
            parent_span_id=ctx.current_span_id if ctx else None,
            span_type=SpanType.NODE,
            name="build_param_constraint",
        )
        ctx.manager.emit(EventType.NODE_PROGRESS, ctx.run_id, span, {
            "agent_id": "constraint",
            "node_id": "build_param_constraint",
            "message": f"取值范围提取完成: {extracted}/{len(params)} 个参数已提取",
            "phase": "range_done",
            "range_count": extracted,
            "params_count": len(params),
            "deterministic_count": deterministic_count,
            "llm_count": len(llm_needed),
        })

    return {"allowed_range_map": result}
