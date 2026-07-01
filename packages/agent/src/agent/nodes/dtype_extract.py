"""DtypeExtract node: extract data types from parameter descriptions via LLM."""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import DTYPE_EXTRACT_PROMPT
from agent.core.llm import create_llm
from agent.utils.llm_common import CONCURRENCY_LIMIT, parse_json_response
from agent.utils.param_validators import VALID_DTYPES, is_cross_reference, is_dash

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


# ---------------------------------------------------------------------------
# Regex fallback & conditional dtype detection (Item 4)
# ---------------------------------------------------------------------------

# dtype token regex covering the VALID_DTYPES whitelist.  Matches
# FLOAT(16|32|64)?, INT(8|16|32|64), UINT(8|16|32|64), BOOL, BFLOAT16,
# BF16, DOUBLE, STRING, COMPLEX(32|64|128).
# Uses ASCII-only lookaround (not \b) so Chinese characters preceding the
# token don't block the match (Python's \b treats CJK as word chars).
_DTYPE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(FLOAT(?:16|32|64)?|INT(?:8|16|32|64)|UINT(?:8|16|32|64)|"
    r"BOOL|BFLOAT16|BF16|DOUBLE|STRING|COMPLEX(?:32|64|128))"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Normalize aliases to canonical VALID_DTYPES spellings.
_DTYPE_ALIASES = {"BF16": "BFLOAT16", "DOUBLE": "FLOAT64"}

# Conditional-dtype (quantization scenario) detection.
# (?<!非) prevents matching "量化" inside "非量化" (non-quantized).
# (?<!non)(?<!non[-.]) prevents matching "quant" inside "non-quant"/"non.quant".
# .*? bridges the keyword and the dtype (handles "mode:", "场景下x为", etc.).
_QUANT_COND_RE = re.compile(
    r"(?:(?<!非)量化|(?<!non)(?<!non[-.])quant\w*)"
    r".*?"
    r"(FLOAT16|INT8|INT32|UINT64|BFLOAT16|BF16|FLOAT32)",
    re.IGNORECASE,
)
_DEFAULT_DTYPE_RE = re.compile(
    r"(?:非量化|不量化|non.quant\w*|普通|正常)"
    r".*?"
    r"(FLOAT16|INT8|INT32|UINT64|BFLOAT16|BF16|FLOAT32)",
    re.IGNORECASE,
)


def _regex_fallback_dtype(description: str) -> str | None:
    """Regex fallback: extract dtype tokens from description text (Item 4).

    Zero LLM cost.  Called when LLM extraction fails, searching the
    ``llm_description`` + ``param_desc`` for VALID_DTYPES whitelist tokens.
    Returns a comma-separated sorted string (e.g. ``"FLOAT16, INT8"``) or
    ``None`` when no token is found.
    """
    if not description:
        return None
    tokens: set[str] = set()
    for m in _DTYPE_TOKEN_RE.finditer(description):
        token = m.group(1).upper()
        token = _DTYPE_ALIASES.get(token, token)
        if token in VALID_DTYPES:
            tokens.add(token)
    if tokens:
        return ", ".join(sorted(tokens))
    return None


def _detect_conditional_dtype(description: str) -> dict | None:
    """Detect conditional dtype (quantization scenario) — Item 4.

    Returns ``{"condition": "量化", "cond_dtype": "INT8",
    "default_dtype": "FLOAT16"}`` or ``None``.
    """
    if not description:
        return None
    quant_m = _QUANT_COND_RE.search(description)
    if not quant_m:
        return None
    cond_dtype = _DTYPE_ALIASES.get(
        quant_m.group(1).upper(), quant_m.group(1).upper()
    )
    if cond_dtype not in VALID_DTYPES:
        return None
    default_dtype = None
    default_m = _DEFAULT_DTYPE_RE.search(description)
    if default_m:
        default_dtype = _DTYPE_ALIASES.get(
            default_m.group(1).upper(), default_m.group(1).upper()
        )
        if default_dtype not in VALID_DTYPES:
            default_dtype = None
    return {
        "condition": "量化",
        "cond_dtype": cond_dtype,
        "default_dtype": default_dtype,
    }


def _is_dtype_valid(dtype_desc: str) -> bool:
    """Check whether an existing dtype_desc value is reasonable.

    Handles JSON format: {"*": "FLOAT16,BFLOAT16"} or {"platform": "FLOAT16"}.
    If any platform value is valid, returns True.
    """
    s = dtype_desc.strip()
    if not s:
        return False
    # Try parsing as JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and parsed:
            return any(
                _is_plain_dtype_valid(v)
                for v in parsed.values()
                if isinstance(v, str)
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return _is_plain_dtype_valid(s)


def _is_plain_dtype_valid(s: str) -> bool:
    """Check a plain text dtype value (non-JSON)."""
    s = s.strip()
    if not s:
        return False
    if is_dash(s):
        return False
    cleaned = s.replace("`", "")
    if is_cross_reference(cleaned):
        return False
    tokens = [t.strip().upper() for t in re.split(r"[,、，/]", s) if t.strip()]
    if not tokens:
        return False
    return all(t in VALID_DTYPES for t in tokens)


async def dtype_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract data type values from parameter descriptions and persist to DB.

    Reads parameters from state (populated by llm_description_extract) instead of
    making a redundant MCP query. Each parameter gets its own LLM call
    for precise extraction, with controlled concurrency.

    Flow:
    1. Read parameters from state.parameters (no MCP query needed)
    2. Filter to parameters with non-empty descriptions
    3. Concurrent LLM call per parameter (Semaphore controlled)
    4. Batch update dtype_desc field via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("DtypeExtract: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("DtypeExtract: no doc_id in state, skipping")
        return {"error": None}

    try:
        params = state.get("parameters", [])
        if not params:
            logger.info("DtypeExtract: no parameters in state for doc_id=%s, skipping", doc_id)
            return {"error": None}

        described = [
            p for p in params
            if p.get("llm_description")
            and (not p.get("dtype_desc") or not _is_dtype_valid(p.get("dtype_desc", "")))
        ]
        if not described:
            logger.info("DtypeExtract: no parameters needing dtype extraction for doc_id=%s, skipping", doc_id)
            return {"error": None}

        # Clear invalid dtype values before re-extraction so downstream
        # nodes don't see stale bad data.
        invalid_clears = [
            {"function_name": p["function_name"], "param_name": p["param_name"], "dtype": ""}
            for p in described
            if p.get("dtype_desc") and not _is_dtype_valid(p.get("dtype_desc", ""))
        ]
        if invalid_clears:
            await _mcp_client.update_param_dtype(doc_id, invalid_clears)
            logger.info(
                "DtypeExtract: cleared %d invalid dtype values (doc_id=%s)",
                len(invalid_clears), doc_id,
            )

        llm = create_llm()
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def _extract_one(param: dict) -> dict | None:
            async with sem:
                return await _extract_dtype(llm, param)

        results = await asyncio.gather(*[_extract_one(p) for p in described])

        # Item 4: Build param_desc map for regex fallback.
        # state.parameters may carry param_desc (set by table_column_extract),
        # but it may be missing for some params.  Query DB to supplement so
        # the regex fallback has the richest text to search.
        param_desc_map: dict[str, str] = {}
        for p in described:
            pn = p.get("param_name", "")
            pd = p.get("param_desc", "") or ""
            if pn and pd:
                param_desc_map[pn] = pd
        if any(not p.get("param_desc") for p in described):
            try:
                db_params = await _mcp_client.query_params_by_doc_id(doc_id)
                for dp in db_params:
                    pn_db = dp.get("param_name", "")
                    if pn_db and pn_db not in param_desc_map:
                        pd_db = dp.get("param_desc", "") or ""
                        if pd_db:
                            param_desc_map[pn_db] = pd_db
            except Exception:
                logger.warning(
                    "DtypeExtract: 查询 param_desc 失败，"
                    "正则兜底仅搜 llm_description",
                )

        # Build LLM result lookup by param_name.
        result_by_name: dict[str, dict | None] = {}
        for r, p in zip(results, described):
            result_by_name[p.get("param_name", "")] = r

        dtype_updates: list[dict] = []
        for p in described:
            pn = p.get("param_name", "")
            fn = p.get("function_name", "")
            r = result_by_name.get(pn)

            combined_desc = " ".join(filter(None, [
                p.get("llm_description", ""),
                param_desc_map.get(pn, ""),
            ]))

            # Prefer LLM result.
            dtype_val = r.get("dtype", "").upper() if r else ""

            # Item 4: LLM failed (empty/invalid) → regex fallback.
            if not dtype_val or not _is_plain_dtype_valid(dtype_val):
                fb = _regex_fallback_dtype(combined_desc)
                if fb:
                    dtype_val = fb
                    logger.info(
                        "DtypeExtract: 正则兜底提取 %s dtype=%s", pn, fb,
                    )

            if not dtype_val:
                continue

            # Item 4: conditional dtype detection (quantization scenario).
            cond = _detect_conditional_dtype(combined_desc)
            if cond and cond["cond_dtype"] != (
                cond.get("default_dtype") or dtype_val
            ):
                default = cond.get("default_dtype") or dtype_val
                dtype_json = {
                    cond["condition"]: cond["cond_dtype"],
                    "*": default,
                }
                dtype_updates.append({
                    "function_name": fn, "param_name": pn,
                    "dtype": json.dumps(dtype_json, ensure_ascii=False),
                })
            else:
                # Wrap plain text dtype as JSON: {"*": value}
                if not dtype_val.startswith("{"):
                    dtype_val = json.dumps(
                        {"*": dtype_val}, ensure_ascii=False,
                    )
                dtype_updates.append({
                    "function_name": fn, "param_name": pn,
                    "dtype": dtype_val,
                })

        if dtype_updates:
            result = await _mcp_client.update_param_dtype(doc_id, dtype_updates)
            logger.info(
                "DtypeExtract: updated dtype for %d/%d parameters (doc_id=%s)",
                result.get("updated", 0),
                len(described),
                doc_id,
            )
        else:
            logger.info("DtypeExtract: no dtypes extracted for doc_id=%s", doc_id)

        return {"error": None}

    except Exception as e:
        logger.exception("DtypeExtract failed for %s", operator_name)
        return {"error": str(e)}


async def _extract_dtype(llm: ChatOpenAI, param: dict) -> dict | None:
    """Call LLM to extract data type for a single parameter."""
    param_name = param.get("param_name", "")
    function_name = param.get("function_name", "")
    description = param.get("llm_description", "")

    prompt = DTYPE_EXTRACT_PROMPT.format(param_name=param_name, params_text=description)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    result = parse_json_response(text, dict)
    if result:
        result["function_name"] = function_name
        if result.get("dtype"):
            result["dtype"] = result["dtype"].upper()
    return result
