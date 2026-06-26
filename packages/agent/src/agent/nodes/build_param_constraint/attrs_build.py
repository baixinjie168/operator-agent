"""AttrsBuild node: assemble deterministic per-parameter attributes.

Handles dtype, format, type, is_optional, is_support_discontinuous,
description, usage_notes, array_length.  No LLM calls.

dtype resolution uses a 3-level fallback chain:
  1. dtype_desc JSON field -> resolve_platform_value per platform
  2. dtype_combinations table -> dtype_by_platform[platform][param_name]
  3. param type -> _ARRAY_TYPE_DTYPE_FALLBACK mapping (e.g. aclIntArray -> "int")

A previous 3rd "legacy data_type -> _split_csv" fallback was removed: the DB
exposes data_type and dtype_desc as aliases of the SAME column, so level 1
already covers it. Worse, applying _split_csv directly to a platform-keyed JSON
string mangles it (splitting on / and 、 chars inside the JSON structure),
producing garbage like ["Atlas A2 ...\": \"FLOAT16", ...]. For platforms
absent from the dtype cell (e.g. null-pointer-only), the empty result is now
preserved instead of being overwritten.

The current 3rd level is different: it derives dtype from the parameter's C
type when levels 1 and 2 both yield nothing. Array types (aclIntArray etc.)
have an inherent primitive dtype; other types (aclTensor, aclScalar, ...) fall
back to the type name itself so downstream generators always receive a
non-empty dtype list.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.nodes.build_param_constraint._helpers import (
    _normalize_type,
    _parse_json_field,
    _split_csv,
)
from agent.nodes.build_param_constraint.state import BuildParamConstraintState
from agent.utils.table_parser import resolve_platform_value

logger = logging.getLogger(__name__)

# Patterns indicating a parameter must be null (empty pointer) on a platform,
# meaning it has no dtype — effectively N/A.
_NULL_POINTER_RE = re.compile(r"(只支持传空指针|传空指针|必须为空指针|仅支持空指针|不支持)")

# Level-3 dtype fallback: when dtype_desc and dtype_combinations both yield
# nothing, derive dtype from the parameter's C type.  Array types have an
# inherent primitive dtype; all other types fall back to the type name itself.
_ARRAY_TYPE_DTYPE_FALLBACK: dict[str, str] = {
    "aclIntArray": "int",
    "aclFloatArray": "float",
    "aclBoolArray": "bool",
}


def _parse_array_length(raw: str) -> dict:
    """Parse the stored array_length column value into {value, src_text}.

    New format (from array_length_extract): a JSON string
    ``{"value": [min, max] | null, "src_text": "..."}``.
    Legacy / non-array: a plain string like "N/A" or an old text description,
    wrapped as ``{"value": <raw>, "src_text": ""}``.
    """
    if isinstance(raw, str) and raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and ("value" in parsed or "src_text" in parsed):
                return {
                    "value": parsed.get("value"),
                    "src_text": parsed.get("src_text", "") or "",
                }
        except (json.JSONDecodeError, TypeError):
            pass
    return {"value": raw, "src_text": ""}


def _is_null_pointer_only(usage_text: str) -> bool:
    """Check if usage_notes text indicates the param must be null on a platform."""
    if not usage_text:
        return False
    return bool(_NULL_POINTER_RE.search(usage_text))


async def attrs_build_node(state: BuildParamConstraintState) -> dict[str, Any]:
    """Assemble deterministic attributes for each (param x platform)."""
    params = state.get("params", [])
    sig_type_map = state.get("sig_type_map", {})
    all_sig_param_names = state.get("all_sig_param_names", [])
    dtype_by_platform = state.get("dtype_by_platform", {})
    supported_platforms = state.get("supported_platforms", [])

    if not params or not supported_platforms:
        return {"attrs_map": {}}

    all_sig_set = set(all_sig_param_names)
    attrs_map: dict[str, dict[str, Any]] = {}

    for param in params:
        pname = param["param_name"]
        fn_name = param["function_name"]

        # Parse platform-aware JSON fields
        shape_json = _parse_json_field(param.get("shape", ""))
        dtype_json = _parse_json_field(param.get("dtype_desc", ""))
        fmt_json = _parse_json_field(param.get("dformat_desc", ""))
        desc_json = _parse_json_field(param.get("param_desc", ""))
        usage_json = _parse_json_field(param.get("usage_notes", ""))

        for plat in supported_platforms:
            # type: sig_type_map -> param_type fallback
            sig_key = f"{fn_name}::{pname}"
            ptype = sig_type_map.get(sig_key, param.get("param_type", ""))
            ptype = _normalize_type(ptype)
            is_tensor = "aclTensor" in ptype

            # usage_notes (needed for dtype N/A detection below)
            usage_raw = resolve_platform_value(usage_json, plat)

            # dtype: 2-level fallback + platform-asymmetric null-pointer handling
            dtype_raw = resolve_platform_value(dtype_json, plat)
            if not dtype_raw:
                # Check if usage_notes indicates "null pointer only" for this
                # platform — if so, the param has no dtype (N/A).
                if usage_raw and _is_null_pointer_only(usage_raw):
                    dtypes = []
                else:
                    dtype_raw_set = dtype_by_platform.get(plat, {}).get(pname, [])
                    if not dtype_raw_set:
                        dtype_raw_set = dtype_by_platform.get("common", {}).get(pname, [])
                    dtypes = sorted(dtype_raw_set) if dtype_raw_set else []

                    # Level 3 fallback: derive dtype from param type when
                    # Level 1 and 2 both yield nothing.  Array types
                    # (aclIntArray etc.) map to their inherent primitive dtype;
                    # other types use the type name itself as a best-effort
                    # fallback so downstream generators always receive a
                    # non-empty dtype list.  Skipped for null-pointer-only
                    # params (handled above) and when ptype itself is empty.
                    if not dtypes and ptype:
                        dtypes = [_ARRAY_TYPE_DTYPE_FALLBACK.get(ptype, ptype)]
            else:
                dtypes = _split_csv(dtype_raw)

            # format: 2-level fallback (mirrors dtype logic)
            #
            # Level 1: dformat_desc JSON field -> resolve_platform_value per platform
            # Level 2: data_format field -> _parse_json_field -> resolve_platform_value
            #
            # dformat_desc and data_format are aliases of the SAME DB column
            # (both map to r[7] in query_params_by_doc_id).  However, in the
            # state pipeline they may be set by different nodes at different
            # times, so level 2 acts as a safety net when level 1 is empty.
            #
            # The old code fed the raw JSON string (e.g. '{"*": "ND"}')
            # directly to _split_csv, producing ['{"*": "ND"}'] instead of
            # ['ND'].  The fix routes the fallback through _parse_json_field
            # + resolve_platform_value so the value is properly extracted
            # before _split_csv.
            if not is_tensor:
                fmt: list | str = "N/A"
            else:
                fmt_raw = resolve_platform_value(fmt_json, plat)
                if not fmt_raw:
                    raw = param.get("data_format", "") or ""
                    if raw:
                        fmt_raw = resolve_platform_value(
                            _parse_json_field(raw), plat
                        )
                fmt = _split_csv(fmt_raw) if fmt_raw else []

            # is_support_discontinuous
            disc_raw = param.get("is_support_discontinuous", "") or ""
            try:
                disc = json.loads(disc_raw) if disc_raw else {"value": "N/A", "src_text": ""}
            except json.JSONDecodeError:
                disc = {"value": disc_raw, "src_text": ""}

            # description
            desc_raw = resolve_platform_value(desc_json, plat)
            if not desc_raw:
                desc_raw = param.get("llm_description", "") or ""

            # shape_raw (preserved for constraint_assemble dimensions src_text)
            shape_raw = resolve_platform_value(shape_json, plat)

            attr_key = f"{fn_name}::{pname}::{plat}"
            attrs_map[attr_key] = {
                "description": desc_raw,
                "usage_notes": {"value": usage_raw, "src_text": ""},
                "type": {"value": ptype, "src_text": ""},
                "format": {"value": fmt, "src_text": ""},
                "is_optional": {"value": bool(param.get("is_optional")), "src_text": ""},
                "is_support_discontinuous": disc,
                "is_operator_param": {"value": pname in all_sig_set, "src_text": ""},
                "array_length": _parse_array_length(param.get("array_length", "N/A") or "N/A"),
                "dtype": {"value": dtypes, "src_text": ""},
                "_shape_raw": shape_raw,
                "_is_tensor": is_tensor,
            }

    logger.info(
        "AttrsBuild: assembled attrs for %d param-platform combos",
        len(attrs_map),
    )
    return {"attrs_map": attrs_map}
