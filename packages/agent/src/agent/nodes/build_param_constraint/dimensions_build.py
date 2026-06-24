"""DimensionsBuild node: parse shape text into structured dimensions arrays.

Deterministic preprocessing (regex) + LLM batch parsing with alignment
validation and per-item fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from agent.core.config import settings
from agent.core.llm import create_llm
from agent.nodes.build_param_constraint._helpers import _parse_json_field
from agent.nodes.build_param_constraint.state import BuildParamConstraintState
from agent.prompts import SHAPE_TO_DIMENSIONS_PROMPT
from agent.utils.llm_common import CONCURRENCY_LIMIT, JSON_BLOCK_RE

logger = logging.getLogger(__name__)



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


def _parse_dimensions_response(text: str) -> list[list] | list[int]:
    """Parse LLM response: either rank spec [N, N] or per-dimension [[min,max], ...]."""
    match = JSON_BLOCK_RE.search(text)
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


async def dimensions_build_node(state: BuildParamConstraintState) -> dict[str, Any]:
    """Collect all shape values and parse via LLM with deterministic preprocessing + validation.

    Flow:
    1. Deterministic preprocessing: regex-based parsing for common patterns
    2. LLM batch parsing for remaining shapes
    3. Alignment validation: if mismatch, fallback to per-item LLM calls
    4. Structure validation: validate each parsed result

    Returns:
        Map of (function_name, param_name, shape_text) → dimensions array.
    """
    params = state.get("params", [])
    if not params:
        return {"dimensions_map": {}}

    shape_entries: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for param in params:
        fn = param.get("function_name", "")
        pn = param.get("param_name", "")
        shape_json = _parse_json_field(param.get("shape", ""))
        # Collect all unique shape values (across all platforms + wildcard)
        for shape_text in shape_json.values():
            shape_text = shape_text.strip() if shape_text else ""
            if shape_text and (fn, pn, shape_text) not in seen:
                seen.add((fn, pn, shape_text))
                shape_entries.append({
                    "function_name": fn,
                    "param_name": pn,
                    "shape": shape_text,
                })

    if not shape_entries:
        return {"dimensions_map": {}}

    # Phase 1: Deterministic preprocessing
    result: dict[str, list] = {}
    llm_needed: list[dict] = []
    deterministic_count = 0

    for entry in shape_entries:
        deterministic = _try_deterministic_parse(entry["shape"])
        if deterministic is not None:
            key = f"{entry['function_name']}::{entry['param_name']}::{entry['shape']}"
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
        return {"dimensions_map": result}

    # Phase 2: LLM batch parsing
    try:
        llm = create_llm()
    except Exception:
        logger.exception("BuildParamConstraint: failed to create LLM for dimensions")
        # Fallback: empty for all remaining
        for entry in llm_needed:
            key = f"{entry['function_name']}::{entry['param_name']}::{entry['shape']}"
            result[key] = []
        return {"dimensions_map": result}

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
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def _process_one(entry: dict) -> tuple[str, list]:
            async with sem:
                single_parsed = await _parse_single(entry)
                is_valid, _ = _validate_dimensions_structure(single_parsed)
                key = f"{entry['function_name']}::{entry['param_name']}::{entry['shape']}"
                return key, single_parsed if is_valid else []

        tasks = [_process_one(e) for e in llm_needed]
        per_item_results = await asyncio.gather(*tasks)
        for key, value in per_item_results:
            result[key] = value
    else:
        # Alignment OK: structure validation + store
        for i, entry in enumerate(llm_needed):
            key = f"{entry['function_name']}::{entry['param_name']}::{entry['shape']}"
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
    return {"dimensions_map": result}
