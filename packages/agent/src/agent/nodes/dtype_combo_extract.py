"""DtypeComboExtract node: extract parameter dtype combinations from constraints section via LLM."""

from __future__ import annotations

import itertools
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import DTYPE_COMBO_TABLE_PROMPT, DTYPE_CONSTRAINT_TEXT_PROMPT
from agent.utils.llm_common import JSON_BLOCK_RE, create_llm, parse_json_response

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_MD_DTYPE_TABLE_RE = re.compile(
    r"^\s*\|[^\n]*数据类型[^\n|]*\|[^\n]*数据类型[^\n|]*\|",
    re.MULTILINE,
)

_MAX_COMBOS_PER_PLATFORM = 500

DTYPE_WHITELIST = {
    "FLOAT32", "FLOAT16", "BFLOAT16",
    "INT4", "INT8", "INT16", "INT32", "INT64",
    "UINT4", "UINT8", "UINT16", "UINT32", "UINT64",
    "BOOL", "DOUBLE",
    "HIFLOAT8", "FLOAT8_E4M3FN", "FLOAT8_E5M2",
    "COMPLEX32", "COMPLEX64",
}

PLATFORM_NAMES = [
    "Atlas 训练系列产品",
    "Atlas 推理系列产品",
    "Atlas A2 训练系列产品/Atlas A2 推理系列产品",
    "Atlas A3 训练系列产品/Atlas A3 推理系列产品",
    "Atlas 200I/500 A2 推理产品",
    "Atlas 350 加速卡",
    "通用",
]


def _get_supported_platforms(state: PipelineState) -> list[str]:
    """Return list of supported platform names from state, falling back to MCP query."""
    products = state.get("product_support", [])
    if products:
        return [
            p.get("product", "")
            for p in products
            if p.get("support", False) and p.get("product", "")
        ]
    return []


def _build_platform_context(supported_platforms: list[str]) -> str:
    """Build platform context string to inject into LLM prompts."""
    if not supported_platforms:
        return ""
    platform_list = "\n".join(f"- {p}" for p in supported_platforms)
    return (
        "适用平台（本算子实际支持的产品列表）：\n"
        f"{platform_list}\n"
        "对于未明确标注平台的约束，platform 字段必须使用上述列表中的名称，不要设为\"通用\"。\n\n"
    )


def _expand_generic_platform(
    combos: list[dict], supported_platforms: list[str]
) -> list[dict]:
    """Expand combos with platform='通用' into one record per supported platform.

    Post-processing safety net: if the LLM still returns '通用' after being
    given the platform context, replace it with the actual supported platforms
    so that downstream code never sees '通用' as a key when specific platforms
    are known.
    """
    if not supported_platforms:
        return combos

    expanded: list[dict] = []
    for combo in combos:
        if combo.get("platform") == "通用":
            for platform in supported_platforms:
                new_combo = dict(combo)
                new_combo["platform"] = platform
                expanded.append(new_combo)
        else:
            expanded.append(combo)
    return expanded


async def dtype_combo_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract parameter dtype combinations from constraints section.

    Flow:
    1. Read state parameters (with dtype_desc, function_name)
    2. Get constraints section via MCP
    3. Determine source: A (combo table) or B (text constraints)
    4. Call LLM + process results
    5. Save dtype_combinations via MCP
    """
    doc_id = state.get("doc_id")
    operator_name = state.get("operator_name")
    if not doc_id or not operator_name:
        logger.warning("dtype_combo_extract: missing doc_id or operator_name, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("dtype_combo_extract: no parameters for doc_id=%s, skipping", doc_id)
            return {"error": None}

        supported_platforms = _get_supported_platforms(state)
        if not supported_platforms:
            # Fall back to MCP query if state doesn't have product_support
            try:
                platform_records = await _mcp_client.query_platform_support_by_doc_id(doc_id)
                supported_platforms = [
                    r.get("platform_name", "")
                    for r in platform_records
                    if r.get("is_supported") and r.get("platform_name", "")
                ]
            except Exception:
                logger.warning(
                    "dtype_combo_extract: could not fetch platform support for doc_id=%s",
                    doc_id,
                )
        platform_context = _build_platform_context(supported_platforms)

        section = await _mcp_client.get_section(doc_id, "constraints")
        if not section or not section.get("content"):
            logger.info("dtype_combo_extract: no constraints section for doc_id=%s, skipping", doc_id)
            return {"error": None}

        constraints_content = section["content"]
        params_text = _build_params_text(params)
        has_combo_table = _has_dtype_combo_table(constraints_content)
        llm = create_llm()

        if has_combo_table:
            combos = await _extract_from_table(
                llm, params, params_text, constraints_content,
                supported_platforms, platform_context,
            )
        else:
            combos = await _extract_from_text(
                llm, params, params_text, constraints_content,
                supported_platforms, platform_context,
            )

        # Post-process: expand "通用" into actual supported platforms
        combos = _expand_generic_platform(combos, supported_platforms)

        if not combos:
            logger.info("dtype_combo_extract: no combos extracted for %s", operator_name)
            return {"error": None}

        result = await _mcp_client.save_dtype_combinations(doc_id, combos)
        logger.info(
            "dtype_combo_extract: saved %d combo records for %s (doc_id=%s)",
            result.get("saved", 0),
            operator_name,
            doc_id,
        )
        return {"error": None}

    except Exception as e:
        logger.exception("dtype_combo_extract failed for %s", operator_name)
        return {"error": str(e)}


def _has_dtype_combo_table(text: str) -> bool:
    """Check if constraints section contains a structured dtype combo table."""
    return bool(_MD_DTYPE_TABLE_RE.search(text))


def _build_params_text(params: list[dict]) -> str:
    """Build parameter summary text for LLM prompt."""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for p in params:
        fn = p.get("function_name", "")
        pn = p.get("param_name", "")
        key = (fn, pn)
        if key in seen:
            continue
        seen.add(key)
        pt = p.get("param_type", "")
        dd = p.get("dtype_desc", "") or ""
        lines.append(f"- {fn}.{pn} (type: {pt}, dtype_desc: {dd})")
    return "\n".join(lines)


async def _extract_from_table(
    llm: ChatOpenAI,
    params: list[dict],
    params_text: str,
    constraints_content: str,
    supported_platforms: list[str],
    platform_context: str,
) -> list[dict]:
    """Extract combos from a structured dtype combo table (Source A)."""
    prompt = DTYPE_COMBO_TABLE_PROMPT.format(
        params_text=params_text,
        table_text=constraints_content,
        platform_context=platform_context,
    )
    response = await llm.ainvoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)

    parsed = _parse_json_array(raw_text)
    if not parsed:
        return []

    combos: list[dict] = []
    for platform_block in parsed:
        platform = platform_block.get("platform", "通用")
        platform = _normalize_platform(platform)
        rows = platform_block.get("rows", [])

        for row in rows:
            expanded = _expand_multi_values(row)
            for combo in expanded:
                fn = _infer_function_name(combo, params)
                combos.append({
                    "function_name": fn,
                    "platform": platform,
                    "combo": combo,
                })

    return combos


def _expand_multi_values(row: dict) -> list[dict]:
    """Expand multi-value cells into separate combos.

    E.g. {"scale": "UINT64/INT64", "x1": "INT8"} ->
         [{"scale": "UINT64", "x1": "INT8"}, {"scale": "INT64", "x1": "INT8"}]
    null values cause the key to be omitted.
    """
    keys: list[str] = []
    value_options: list[list[str]] = []

    for k, v in row.items():
        v_str = str(v).strip() if v is not None else ""
        if v_str.lower() == "null" or v_str == "":
            continue
        if "/" in v_str:
            keys.append(k)
            value_options.append([opt.strip() for opt in v_str.split("/")])
        else:
            keys.append(k)
            value_options.append([v_str])

    if not keys:
        return [{}]

    total = 1
    for opts in value_options:
        total *= len(opts)
    if total > _MAX_COMBOS_PER_PLATFORM:
        logger.warning(
            "dtype_combo_extract: multi-value expansion too large (%d), truncating",
            total,
        )
        return [{k: opts[0] for k, opts in zip(keys, value_options, strict=True)}]

    results: list[dict] = []
    for combo_vals in itertools.product(*value_options):
        combo = dict(zip(keys, combo_vals, strict=True))
        results.append(combo)

    return results


def _generate_combos_from_constraints(
    constraints: list[dict],
    dtype_desc_map: dict[str, set[str]],
    function_name: str,
) -> list[dict]:
    """Generate combo dicts from extracted text constraints with coupling detection."""
    positive: dict[str, list[str]] = {}
    negative: dict[str, list[str]] = {}

    for c in constraints:
        pn = c.get("param_name", "")
        mode = c.get("mode", "")
        dtypes = [d.upper() for d in c.get("dtypes", []) if d.upper() in DTYPE_WHITELIST]

        if mode == "positive":
            positive[pn] = dtypes
        elif mode == "negative":
            negative[pn] = dtypes

    all_params = set(positive.keys()) | set(negative.keys())
    resolved: dict[str, list[str]] = {}

    for pn in all_params:
        if pn in positive:
            resolved[pn] = positive[pn]
        elif pn in negative:
            full_set = dtype_desc_map.get(pn, set())
            if not full_set:
                logger.warning(
                    "dtype_combo_extract: no dtype_desc for %s.%s, skipping negative constraint",
                    function_name, pn,
                )
                continue
            resolved[pn] = sorted(full_set - set(negative[pn]))

    if not resolved:
        return []

    # Coupling detection: group params with identical dtype lists
    groups: list[tuple[list[str], list[str]]] = []
    seen: set[str] = set()
    for pn, dtypes in resolved.items():
        if pn in seen:
            continue
        coupled = [p for p, d in resolved.items() if d == dtypes and p not in seen]
        for p in coupled:
            seen.add(p)
        groups.append((coupled, dtypes))

    # Each coupling group varies together
    group_options: list[list[dict]] = []
    for coupled_params, dtypes in groups:
        options: list[dict] = []
        for dt in dtypes:
            option = {pn: dt for pn in coupled_params}
            options.append(option)
        group_options.append(options)

    total = 1
    for opts in group_options:
        total *= len(opts)
    if total > _MAX_COMBOS_PER_PLATFORM:
        logger.warning(
            "dtype_combo_extract: coupling expansion too large (%d), truncating",
            total,
        )
        combo: dict[str, str] = {}
        for opts in group_options:
            combo.update(opts[0])
        return [combo]

    results: list[dict] = []
    for combo_tuple in itertools.product(*group_options):
        merged: dict[str, str] = {}
        for opt in combo_tuple:
            merged.update(opt)
        results.append(merged)

    return results


async def _extract_from_text(
    llm: ChatOpenAI,
    params: list[dict],
    params_text: str,
    constraints_content: str,
    supported_platforms: list[str],
    platform_context: str,
) -> list[dict]:
    """Extract combos from text-based dtype constraints (Source B)."""
    prompt = DTYPE_CONSTRAINT_TEXT_PROMPT.format(
        params_text=params_text,
        constraints_text=constraints_content,
        platform_context=platform_context,
    )
    response = await llm.ainvoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)

    constraints = _parse_json_array(raw_text)
    if not constraints:
        return []

    dtype_desc_map = _build_dtype_desc_map(params)

    # Group constraints by (function_name, platform)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for c in constraints:
        param_name = c.get("param_name", "")
        platform = _normalize_platform(c.get("platform", "通用"))
        fn = _find_function_for_param(param_name, params)
        key = (fn, platform)
        grouped.setdefault(key, []).append(c)

    combos: list[dict] = []
    for (fn, platform), group in grouped.items():
        generated = _generate_combos_from_constraints(group, dtype_desc_map, fn)
        for combo in generated:
            combos.append({
                "function_name": fn,
                "platform": platform,
                "combo": combo,
            })

    return combos


# ── Helpers ──


def _normalize_platform(platform: str) -> str:
    """Normalize platform name to standard form."""
    p = platform.strip()
    p = re.sub(r"<[^>]+>", "", p).strip()
    if not p:
        return "通用"
    for name in PLATFORM_NAMES:
        if name == p or name in p or p in name:
            return name
    return p


def _parse_json_array(text: str) -> list[dict]:
    """Extract JSON array from LLM response."""
    text = text.strip()
    match = JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    logger.warning("dtype_combo_extract: failed to parse LLM response: %s", text[:200])
    return []


def _build_dtype_desc_map(params: list[dict]) -> dict[str, set[str]]:
    """Build param_name -> set of dtype values from dtype_desc field."""
    result: dict[str, set[str]] = {}
    for p in params:
        pn = p.get("param_name", "")
        dd = p.get("dtype_desc", "") or ""
        if not dd:
            continue
        dtypes = set()
        for part in re.split(r"[、，,/]", dd):
            part = part.strip().upper()
            if part in DTYPE_WHITELIST:
                dtypes.add(part)
        if dtypes:
            result[pn] = dtypes
    return result


def _find_function_for_param(param_name: str, params: list[dict]) -> str:
    """Find the function_name for a given param_name."""
    for p in params:
        if p.get("param_name") == param_name:
            return p.get("function_name", "")
    return ""


def _infer_function_name(combo: dict, params: list[dict]) -> str:
    """Infer function_name from combo keys by matching against params list."""
    for p in params:
        if p.get("param_name") in combo:
            return p.get("function_name", "")
    return ""