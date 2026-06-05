"""BuildParamConstraint node: assemble structured param_constraint JSON for each parameter."""

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
                    dtype_desc = param.get("dtype_desc", "") or ""
                    if dtype_desc:
                        dtypes = sorted({
                            d.strip() for d in re.split(r"[、，,/]", dtype_desc)
                            if d.strip()
                        })

                # format: split by "/"
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
                is_tensor = "aclTensor" in ptype
                if is_tensor:
                    ar_value: list = []
                else:
                    ar_value = ar_map.get((fn_name, pname), [])

                constraint[plat] = {
                    "description": param.get("param_desc", "") or "",
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


async def _batch_parse_dimensions(params: list[dict]) -> dict[tuple[str, str], list]:
    """Collect all shape values and parse via LLM in one batch call.

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

    # Build indexed list for LLM
    indexed_shapes = [
        f"{i + 1}. {e['shape']}" for i, e in enumerate(shape_entries)
    ]
    shapes_text = "\n".join(indexed_shapes)

    try:
        llm = ChatOpenAI(
            api_key=settings.active_api_key,
            base_url=settings.active_base_url,
            model=settings.active_model,
            temperature=0.1,
        )
        prompt = SHAPE_TO_DIMENSIONS_PROMPT.format(shapes=shapes_text)
        response = await llm.ainvoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        parsed = _parse_dimensions_response(text)

        result: dict[tuple[str, str], list] = {}
        for i, entry in enumerate(shape_entries):
            key = (entry["function_name"], entry["param_name"])
            if i < len(parsed):
                result[key] = parsed[i]
            else:
                result[key] = []

        logger.info("BuildParamConstraint: parsed %d dimensions via LLM", len(result))
        return result

    except Exception:
        logger.exception("BuildParamConstraint: LLM dimensions parsing failed")
        return {}


async def _batch_extract_allowed_range(
    params: list[dict],
    constraints_text: str,
) -> dict[tuple[str, str], list]:
    """Batch extract allowed_range_value for non-Tensor params via LLM.

    Uses per-platform extraction: each (param, platform) gets its own LLM call,
    but params without platform-specific context share a single call.

    Returns:
        Map of (function_name, param_name) → [[min,max], ...] array.
    """
    if not params:
        return {}

    result: dict[tuple[str, str], list] = {}
    sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

    try:
        llm = ChatOpenAI(
            api_key=settings.active_api_key,
            base_url=settings.active_base_url,
            model=settings.active_model,
            temperature=0.1,
        )
    except Exception:
        logger.exception("BuildParamConstraint: failed to create LLM for allowed_range")
        return {}

    async def _extract_one(param: dict) -> tuple[tuple[str, str], list]:
        async with sem:
            key = (param["function_name"], param["param_name"])
            c_type = param.get("param_type", "")
            param_desc = param.get("param_desc", "") or ""

            context_parts: list[str] = []
            if constraints_text:
                context_parts.append(f"## 约束说明\n{constraints_text}")
            if param_desc:
                context_parts.append(f"## 参数使用说明\n{param_desc}")

            context_text = "\n\n".join(context_parts) if context_parts else ""
            if not context_text.strip():
                return key, []

            try:
                prompt = ALLOWED_RANGE_VALUE_BUILD_PROMPT.format(
                    param_name=param["param_name"],
                    param_type=c_type,
                    context_text=context_text,
                )
                response = await llm.ainvoke(prompt)
                text = response.content if hasattr(response, "content") else str(response)
                parsed = _parse_allowed_range_response(text)
                return key, parsed
            except Exception:
                logger.warning(
                    "BuildParamConstraint: LLM allowed_range failed for %s",
                    param["param_name"],
                )
                return key, []

    results = await asyncio.gather(*[_extract_one(p) for p in params])
    for key, value in results:
        result[key] = value

    extracted = sum(1 for v in result.values() if v)
    logger.info(
        "BuildParamConstraint: extracted allowed_range for %d/%d non-Tensor params",
        extracted, len(params),
    )
    return result


def _parse_dimensions_response(text: str) -> list[list]:
    """Parse LLM response: array of dimension arrays."""
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

    # Regex fallback for outer array
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
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
