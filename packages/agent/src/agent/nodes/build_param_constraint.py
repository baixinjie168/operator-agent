"""BuildParamConstraint node: assemble structured param_constraint JSON for each parameter.

Implements deterministic preprocessing + validation for dimensions and allowed_range_value:
- Phase 1: Deterministic preprocessing (regex-based, zero LLM cost)
- Phase 1: Structural validation (dual format for dimensions, min/max/alignment for ranges)
- Phase 2: Failure remediation (per-item retry on alignment failure, retry on validation failure)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import ALLOWED_RANGE_VALUE_BUILD_PROMPT, SHAPE_TO_DIMENSIONS_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_CONCURRENCY_LIMIT = 5


async def build_param_constraint_node(state: PipelineState) -> dict[str, Any]:
    """Build param_constraint JSON for each parameter and persist to DB.

    Flow:
    1. Query params (full columns), function_signatures, platform_support, dtype_combos
    2. Build indexes: sig_type_map, operator_params, dtype_by_platform
    3. LLM batch parse shape → dimensions
    4. LLM batch extract allowed_range_value (non-Tensor params only)
    5. Assemble constraint JSON per (function_name, param_name) × platform
    6. Batch update param_constraint via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("BuildParamConstraint: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("BuildParamConstraint: no doc_id, skipping")
        return {"error": None}

    try:
        # Step 1: Query all data sources
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        sigs = await _mcp_client.query_function_signatures_by_doc_id(doc_id)
        platforms = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        dtype_combos = await _mcp_client.query_dtype_combos_by_doc_id(doc_id)

        if not params:
            logger.info("BuildParamConstraint: no parameters, skipping")
            return {"error": None}

        # Step 2: Build indexes
        sig_type_map: dict[tuple[str, str], str] = {}
        operator_params: set[str] = set()
        for sig in sigs:
            for p in sig.get("parameters", []):
                sig_type_map[(sig["function_name"], p["name"])] = p.get("type", "")
                if not sig["function_name"].endswith("GetWorkspaceSize"):
                    operator_params.add(p["name"])

        dtype_by_platform: dict[str, dict[str, set[str]]] = {}
        for combo in dtype_combos:
            plat = combo.get("platform", "通用")
            dtype_by_platform.setdefault(plat, {})
            for pname, dtype_val in combo.get("combo", {}).items():
                dtype_by_platform[plat].setdefault(pname, set())
                if isinstance(dtype_val, str) and "/" in dtype_val:
                    for d in dtype_val.split("/"):
                        dtype_by_platform[plat][pname].add(d.strip())
                else:
                    dtype_by_platform[plat][pname].add(str(dtype_val))

        supported_platforms = [
            p["platform_name"] for p in platforms if p.get("is_supported") == 1
        ]

        # Step 3: LLM batch parse shape → dimensions
        shape_map = await _batch_parse_dimensions(params)

        # Step 4: LLM batch extract allowed_range_value for non-Tensor params
        non_tensor_params = [
            p for p in params
            if "aclTensor" not in sig_type_map.get(
                (p["function_name"], p["param_name"]), ""
            )
        ]
        constraints_section = await _mcp_client.get_section(doc_id, "constraints")
        constraints_text = (constraints_section or {}).get("content", "") or ""
        ar_map = await _batch_extract_allowed_range(
            non_tensor_params, constraints_text
        )

        # Step 5: Assemble constraint JSON for each parameter
        updates: list[dict] = []
        for param in params:
            pname = param["param_name"]
            fn_name = param["function_name"]
            constraint: dict[str, Any] = {}

            for plat in supported_platforms:
                ptype = sig_type_map.get((fn_name, pname), param.get("param_type", ""))
                # Strip const and pointer modifiers (defensive normalization)
                ptype = _normalize_type(ptype)

                # dtype: platform-specific → "通用" → dtype_desc fallback
                dtypes = sorted(
                    dtype_by_platform.get(plat, {}).get(pname, set())
                )
                if not dtypes:
                    dtypes = sorted(
                        dtype_by_platform.get("通用", {}).get(pname, set())
                    )
                if not dtypes:
                    # Fallback: parse dtype_desc (e.g. "FLOAT32,FLOAT16,BFLOAT16")
                    dtype_desc = param.get("data_type", "") or ""
                    if dtype_desc:
                        dtypes = sorted({
                            d.strip() for d in re.split(r"[、，,/]", dtype_desc)
                            if d.strip()
                        })

                # format: non-Tensor → "N/A", Tensor → array (empty or split by "/")
                is_tensor = "aclTensor" in ptype
                if not is_tensor:
                    fmt: list | str = "N/A"
                else:
                    fmt_str = param.get("data_format", "") or ""
                    fmt = [f.strip() for f in fmt_str.split("/") if f.strip()]

                # is_support_discontinuous: JSON parse
                disc_raw = param.get("is_support_discontinuous", "") or ""
                try:
                    disc = json.loads(disc_raw) if disc_raw else {"value": "N/A", "src_text": ""}
                except json.JSONDecodeError:
                    disc = {"value": disc_raw, "src_text": ""}

                # dimensions: from LLM result
                shape_raw = param.get("shape", "") or ""
                dimensions_value = shape_map.get((fn_name, pname), [])

                # allowed_range_value: Tensor → [], else from LLM
                if is_tensor:
                    ar_value: list = []
                else:
                    ar_value = ar_map.get((fn_name, pname), [])

                constraint[plat] = {
                    "description": param.get("param_desc", "") or param.get("llm_description", "") or "",
                    "type": {"value": ptype, "src_text": ""},
                    "format": {"value": fmt, "src_text": ""},
                    "is_optional": {"value": bool(param.get("is_optional")), "src_text": ""},
                    "is_support_discontinuous": disc,
                    "is_operator_param": {
                        "value": pname in operator_params,
                        "src_text": "",
                    },
                    "dimensions": {"value": dimensions_value, "src_text": shape_raw},
                    "array_length": param.get("array_length", "N/A") or "N/A",
                    "dtype": {"value": dtypes, "src_text": ""},
                    "allowed_range_value": {"value": ar_value, "src_text": ""},
                }

            updates.append({
                "function_name": fn_name,
                "param_name": pname,
                "param_constraint": json.dumps(constraint, ensure_ascii=False),
            })

        # Step 6: Batch update
        if updates:
            result = await _mcp_client.update_param_constraint(doc_id, updates)
            logger.info(
                "BuildParamConstraint: updated %d/%d params (doc_id=%s)",
                result.get("updated", 0), len(updates), doc_id,
            )

        return {"error": None}

    except Exception as e:
        logger.exception("BuildParamConstraint failed for %s", operator_name)
        return {"error": str(e)}


def _is_bool_type(param_type: str) -> bool:
    """Check if parameter type is bool."""
    return param_type.lower() == "bool"


def _normalize_type(ptype: str) -> str:
    """Strip const, pointer *, and reference & from C type names.

    Defensive normalization ensuring param_constraint.type.value contains
    only the base type name (e.g. "aclTensor" not "const aclTensor *").
    """
    ptype = re.sub(r'\bconst\b', '', ptype)
    ptype = ptype.replace('*', '').replace('&', '').strip()
    return ptype


# ---------------------------------------------------------------------------
# Phase 1: Deterministic preprocessing for dimensions (zero LLM cost)
# ---------------------------------------------------------------------------

# Pattern: (regex, result_or_lambda)
# Result format:
#   [min_rank, max_rank] for rank/rank-range (e.g. [5,5] exact, [0,8] range)
#   [[min,max], ...] for per-dimension size ranges
_DIMENSION_PATTERNS: list[tuple[str, Any]] = [
    # "标量" / "0-D" / "0D" → [] (scalar has no dimensions)
    (r"^(标量|0[- ]?D|0D)$", []),
    # "X-Y" / "X~Y" rank range (e.g. "0-8" → [0, 8], "1-8" → [1, 8])
    # Must come before N-D to avoid "1-8" being partially consumed
    (
        r"^(\d+)\s*[-~]\s*(\d+)$",
        lambda m: [int(m.group(1)), int(m.group(2))],
    ),
    # "1-D" / "1D" → [1, 1] (rank format)
    (r"^1[- ]?D$", [1, 1]),
    # "ND" / "N-D" → [N, N] (rank format)
    (r"^(\d+)[- ]?D$", lambda m: [int(m.group(1)), int(m.group(1))]),
    # "(N,C,H,W)" → count dimensions → [4, 4] (rank format)
    (r"^\(([^)]+)\)$", lambda m: [len(m.group(1).split(","))] * 2),
    # "[2, 3, 4]" → fixed dimensions → [[2,2],[3,3],[4,4]] (per-dimension format)
    (
        r"^\[([^\]]+)\]$",
        lambda m: [
            [int(v.strip()), int(v.strip())]
            for v in m.group(1).split(",")
            if v.strip().isdigit()
        ],
    ),
    # "与输入相同" / "same as input" → [] (cannot determine)
    (r"^(与输入相同|同输入|same as input)$", []),
]


def _try_deterministic_parse(shape: str) -> list | None:
    """Try deterministic parsing of shape string.

    Returns parsed result if pattern matches, None otherwise.
    """
    shape_stripped = shape.strip()
    for pattern, result in _DIMENSION_PATTERNS:
        m = re.match(pattern, shape_stripped, re.IGNORECASE)
        if m:
            if callable(result):
                return result(m)
            return result
    return None


# ---------------------------------------------------------------------------
# Phase 1: Validation functions for dimensions
# ---------------------------------------------------------------------------


def _is_rank_format(dims: list) -> bool:
    """Check if dimensions is rank format [count, count]."""
    return (
        isinstance(dims, list)
        and len(dims) == 2
        and all(isinstance(d, int) for d in dims)
    )


def _validate_dimensions_structure(dims: list) -> tuple[bool, str]:
    """Validate structure of dimensions array (supports three formats).

    Format 1 (rank): [min_rank, max_rank] where 0 <= min_rank <= max_rank <= 10
    Format 2 (per-dimension): [[min, max], ...] where min <= max or null
    """
    if not isinstance(dims, list):
        return False, "dimensions must be a list"

    if not dims:  # Empty array is valid (scalar or undetermined)
        return True, ""

    # Format 1: Rank format [min_rank, max_rank]
    if _is_rank_format(dims):
        min_rank, max_rank = dims[0], dims[1]
        if min_rank < 0:
            return False, f"rank min must be >= 0, got {min_rank}"
        if min_rank > max_rank:
            return False, f"rank [min, max] requires min <= max, got {dims}"
        if max_rank > 10:
            return False, f"Too many dimensions: {max_rank}"
        return True, ""

    # Format 2: Per-dimension ranges [[min, max], ...]
    for i, dim in enumerate(dims):
        if not isinstance(dim, list) or len(dim) != 2:
            return False, f"dim[{i}] must be [min, max], got {dim}"

        min_val, max_val = dim
        if min_val is not None and max_val is not None:
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                return False, f"dim[{i}] values must be int/float or null, got [{type(min_val).__name__}, {type(max_val).__name__}]"
            if min_val > max_val:
                return False, f"dim[{i}]: min ({min_val}) > max ({max_val})"

    if len(dims) > 10:
        return False, f"Too many dimensions: {len(dims)}"

    return True, ""


def _validate_dimensions_alignment(
    input_count: int,
    parsed: list,
) -> tuple[bool, str]:
    """Validate that parsed array length matches input count."""
    if len(parsed) != input_count:
        return False, f"Alignment mismatch: expected {input_count} entries, got {len(parsed)}"
    return True, ""


# ---------------------------------------------------------------------------
# Phase 1: Deterministic preprocessing for allowed_range_value
# ---------------------------------------------------------------------------

# Pattern: (regex, result_or_lambda)
_RANGE_PATTERNS: list[tuple[str, Any]] = [
    # "枚举值: 1, 2, 3" → [[1,1], [2,2], [3,3]] (must come before range pattern)
    (
        r"枚举值\s*[:：]\s*(.+)",
        lambda m: [
            [int(v.strip()), int(v.strip())]
            for v in m.group(1).split(",")
            if v.strip().lstrip("-").isdigit()
        ],
    ),
    # "范围0-100" / "0~100" / "[0, 100]" → [[0, 100]]
    (
        r"\[?\s*(-?\d+)\s*[,，\-~]\s*(-?\d+)\s*\]?",
        lambda m: [[int(m.group(1)), int(m.group(2))]],
    ),
]


def _try_deterministic_range(text: str) -> list | None:
    """Try deterministic extraction of range from text.

    Returns parsed result if pattern matches, None otherwise.
    """
    for pattern, extractor in _RANGE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            try:
                result = extractor(m)
                if result:  # Only return if non-empty
                    return result
            except (ValueError, IndexError):
                continue
    return None


# ---------------------------------------------------------------------------
# Phase 1: Validation functions for allowed_range_value
# ---------------------------------------------------------------------------


def _validate_range_structure(
    ranges: list,
    param_type: str = "",
) -> tuple[bool, str]:
    """Validate structure of allowed_range_value array.

    Checks:
    1. Each element is [min, max] with min <= max
    2. Values are int/float or null
    3. Type compatibility (unsigned types must be non-negative)
    4. Reasonable values (not exceeding 10^9)
    """
    if not isinstance(ranges, list):
        return False, "ranges must be a list"

    if not ranges:  # Empty is valid
        return True, ""

    for i, r in enumerate(ranges):
        if not isinstance(r, list) or len(r) != 2:
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


async def _batch_parse_dimensions(params: list[dict]) -> dict[tuple[str, str], list[list] | list[int]]:
    """Collect all shape values and parse via LLM with deterministic preprocessing + validation.

    Flow:
    1. Deterministic preprocessing: regex-based parsing for common patterns
    2. LLM batch parsing for remaining shapes
    3. Alignment validation: if mismatch, fallback to per-item LLM calls
    4. Structure validation: validate each parsed result

    Returns:
        Map of (function_name, param_name) → dimensions array.
    """
    shape_entries: list[dict] = []
    for param in params:
        shape_raw = (param.get("shape", "") or "").strip()
        if shape_raw:
            shape_entries.append({
                "function_name": param.get("function_name", ""),
                "param_name": param.get("param_name", ""),
                "shape": shape_raw,
            })

    if not shape_entries:
        return {}

    # Phase 1: Deterministic preprocessing
    result: dict[tuple[str, str], list] = {}
    llm_needed: list[dict] = []
    deterministic_count = 0

    for entry in shape_entries:
        deterministic = _try_deterministic_parse(entry["shape"])
        if deterministic is not None:
            key = (entry["function_name"], entry["param_name"])
            is_valid, _ = _validate_dimensions_structure(deterministic)
            result[key] = deterministic if is_valid else []
            deterministic_count += 1
        else:
            llm_needed.append(entry)

    if deterministic_count > 0:
        logger.info(
            "BuildParamConstraint: deterministic preprocessing handled %d/%d shapes",
            deterministic_count, len(shape_entries),
        )

    if not llm_needed:
        return result

    # Phase 2: LLM batch parsing
    try:
        llm = ChatOpenAI(
            api_key=settings.active_api_key,
            base_url=settings.active_base_url,
            model=settings.active_model,
            temperature=0.1,
        )
    except Exception:
        logger.exception("BuildParamConstraint: failed to create LLM for dimensions")
        # Fallback: empty for all remaining
        for entry in llm_needed:
            key = (entry["function_name"], entry["param_name"])
            result[key] = []
        return result

    async def _parse_single(entry: dict) -> list:
        """Parse a single shape via LLM."""
        prompt = SHAPE_TO_DIMENSIONS_PROMPT.format(shapes=f"1. {entry['shape']}")
        try:
            response = await llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            parsed = _parse_dimensions_response(text)
            if parsed and len(parsed) == 1:
                return parsed[0]
            return parsed if isinstance(parsed, list) and _is_rank_format(parsed) else []
        except Exception:
            logger.warning("BuildParamConstraint: LLM failed for shape '%s'", entry["shape"])
            return []

    # Build indexed list for batch LLM call
    indexed_shapes = [f"{i + 1}. {e['shape']}" for i, e in enumerate(llm_needed)]
    shapes_text = "\n".join(indexed_shapes)

    parsed: list = []
    for attempt in range(settings.dimensions_max_retries + 1):
        try:
            prompt = SHAPE_TO_DIMENSIONS_PROMPT.format(shapes=shapes_text)
            response = await llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            parsed = _parse_dimensions_response(text)
            break
        except Exception:
            logger.warning(
                "BuildParamConstraint: LLM batch call failed (attempt %d/%d)",
                attempt + 1, settings.dimensions_max_retries + 1,
            )
            if attempt == settings.dimensions_max_retries:
                parsed = []

    # Phase 3: Alignment validation
    is_aligned, error = _validate_dimensions_alignment(len(llm_needed), parsed)

    if not is_aligned:
        logger.warning(
            "BuildParamConstraint: dimensions alignment failed: %s — fallback to per-item",
            error,
        )
        # Fallback: per-item LLM calls
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _process_one(entry: dict) -> tuple[tuple[str, str], list]:
            async with sem:
                single_parsed = await _parse_single(entry)
                is_valid, _ = _validate_dimensions_structure(single_parsed)
                key = (entry["function_name"], entry["param_name"])
                return key, single_parsed if is_valid else []

        tasks = [_process_one(e) for e in llm_needed]
        per_item_results = await asyncio.gather(*tasks)
        for key, value in per_item_results:
            result[key] = value
    else:
        # Alignment OK: structure validation + store
        for i, entry in enumerate(llm_needed):
            key = (entry["function_name"], entry["param_name"])
            dims = parsed[i]
            is_valid, validation_error = _validate_dimensions_structure(dims)
            if is_valid:
                result[key] = dims
            else:
                logger.warning(
                    "BuildParamConstraint: dimensions structure invalid for %s.%s: %s",
                    entry["function_name"], entry["param_name"], validation_error,
                )
                result[key] = []

    logger.info(
        "BuildParamConstraint: parsed %d dimensions (%d deterministic, %d LLM)",
        len(result), deterministic_count, len(llm_needed),
    )
    return result


async def _batch_extract_allowed_range(
    params: list[dict],
    constraints_text: str,
) -> dict[tuple[str, str], list]:
    """Batch extract allowed_range_value for non-Tensor params via LLM.

    Flow:
    1. Bool type params: short-circuit with [True, False]
    2. Deterministic preprocessing: regex-based extraction for common patterns
    3. LLM extraction with retry for remaining params
    4. Structural validation: min <= max, type compatibility
    5. Source validation: check if values appear in source text (soft check)

    Returns:
        Map of (function_name, param_name) → [[min,max], ...] array.
    """
    if not params:
        return {}

    result: dict[tuple[str, str], list] = {}
    sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

    # Phase 0: Separate bool params (short-circuit)
    bool_params: list[dict] = []
    remaining_params: list[dict] = []
    for p in params:
        if _is_bool_type(p.get("param_type", "")):
            bool_params.append(p)
        else:
            remaining_params.append(p)

    for p in bool_params:
        key = (p["function_name"], p["param_name"])
        result[key] = [True, False]

    # Phase 1: Deterministic preprocessing
    llm_needed: list[dict] = []
    deterministic_count = 0

    for p in remaining_params:
        llm_desc = p.get("llm_description", "") or ""
        if not llm_desc.strip() and not constraints_text.strip():
            key = (p["function_name"], p["param_name"])
            result[key] = []
            continue

        deterministic = _try_deterministic_range(llm_desc)
        if deterministic is None:
            deterministic = _try_deterministic_range(constraints_text)

        if deterministic is not None:
            key = (p["function_name"], p["param_name"])
            param_type = p.get("param_type", "")
            is_valid, _ = _validate_range_structure(deterministic, param_type)
            result[key] = deterministic if is_valid else []
            deterministic_count += 1
        else:
            llm_needed.append(p)

    if deterministic_count > 0:
        logger.info(
            "BuildParamConstraint: deterministic range extraction handled %d/%d params",
            deterministic_count, len(remaining_params),
        )

    if not llm_needed:
        return result

    # Phase 2: LLM extraction with retry + validation
    try:
        llm = ChatOpenAI(
            api_key=settings.active_api_key,
            base_url=settings.active_base_url,
            model=settings.active_model,
            temperature=0.1,
        )
    except Exception:
        logger.exception("BuildParamConstraint: failed to create LLM for allowed_range")
        for p in llm_needed:
            key = (p["function_name"], p["param_name"])
            result[key] = []
        return result

    async def _extract_one_with_retry(param: dict) -> tuple[tuple[str, str], list]:
        async with sem:
            key = (param["function_name"], param["param_name"])
            c_type = param.get("param_type", "")
            llm_desc = param.get("llm_description", "") or ""

            context_parts: list[str] = []
            if constraints_text:
                context_parts.append(f"## 约束说明\n{constraints_text}")
            if llm_desc:
                context_parts.append(f"## 参数使用说明\n{llm_desc}")

            context_text = "\n\n".join(context_parts) if context_parts else ""
            if not context_text.strip():
                return key, []

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

                    # Structural validation
                    is_valid, error = _validate_range_structure(parsed, c_type)
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
                            return key, []

                    # Source validation (soft check)
                    _validate_range_source(parsed, context_text)

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
                        return key, []

            return key, []

    tasks = [_extract_one_with_retry(p) for p in llm_needed]
    llm_results = await asyncio.gather(*tasks)
    for key, value in llm_results:
        result[key] = value

    extracted = sum(1 for v in result.values() if v)
    logger.info(
        "BuildParamConstraint: extracted allowed_range for %d/%d non-Tensor params "
        "(%d bool short-circuited, %d deterministic, %d LLM)",
        extracted, len(params), len(bool_params), deterministic_count, len(llm_needed),
    )
    return result


def _parse_dimensions_response(text: str) -> list[list] | list[int]:
    """Parse LLM response: either rank spec [N, N] or per-dimension [[min,max], ...]."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            # Rank specification: flat list of ints like [5, 5]
            if data and all(isinstance(item, int) for item in data):
                return data
            # Per-dimension ranges: nested lists like [[null,null], [3,3]]
            return [item if isinstance(item, list) else [] for item in data]
    except json.JSONDecodeError:
        pass

    # Regex fallback for outer array
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                # Rank specification: flat list of ints like [5, 5]
                if data and all(isinstance(item, int) for item in data):
                    return data
                # Per-dimension ranges: nested lists like [[null,null], [3,3]]
                return [item if isinstance(item, list) else [] for item in data]
        except json.JSONDecodeError:
            pass

    logger.warning("BuildParamConstraint: failed to parse dimensions: %s", text[:200])
    return []


def _parse_allowed_range_response(text: str) -> list:
    """Parse LLM response: [[min,max], ...] array."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [item if isinstance(item, list) else [] for item in data]
    except json.JSONDecodeError:
        pass

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                return [item if isinstance(item, list) else [] for item in data]
        except json.JSONDecodeError:
            pass

    logger.warning("BuildParamConstraint: failed to parse allowed_range: %s", text[:200])
    return []
