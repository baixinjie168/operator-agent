"""ValidateResults node: attribute checklist verification + coverage report.

Runs after both ``extract_ws`` and ``extract_exe`` have completed. For each
extracted parameter, checks whether the ``llm_description`` covers all
attributes that are present in the original context.
"""

from __future__ import annotations

import json
import logging
import re as _re
from typing import Any

from agent.nodes.llm_description_extract.state import DescriptionExtractState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribute checklist
# ---------------------------------------------------------------------------

def _section_filled(desc: str, header: str) -> bool:
    """Check whether a structured ``# header`` section in *desc* has real content.

    Returns True when the section exists and its body is not just "无" or
    whitespace.  For backward compatibility with old natural-language
    descriptions (which never contain ``# `` headers), returns False so
    that the caller can fall back to keyword-based checks.
    """
    pattern = _re.compile(
        r"^#\s+" + _re.escape(header) + r"\s*\n(.*?)(?=^#|\Z)",
        _re.MULTILINE | _re.DOTALL,
    )
    m = pattern.search(desc)
    if m is None:
        return False
    body = m.group(1).strip()
    return bool(body) and body != "无"


def _has_json_dtype(result: dict) -> bool:
    """Check if dtype_desc field contains JSON values.

    When a parameter's dtype is entirely platform-specific (e.g. all values
    tagged with <term>PLATFORM</term>), the LLM correctly omits it from
    llm_description. But the dtype IS captured by table_column_extract
    in the dtype_desc JSON field. This check prevents false-positive warnings.
    """
    try:
        dtype_raw = result.get("dtype_desc", "") or ""
        if not dtype_raw:
            return False
        parsed = json.loads(dtype_raw) if isinstance(dtype_raw, str) else dtype_raw
        if isinstance(parsed, dict):
            return any(v for v in parsed.values() if isinstance(v, str) and v.strip())
    except (json.JSONDecodeError, TypeError):
        pass
    return False


ATTR_CHECKLIST: list[tuple[str, Any]] = [
    ("direction", lambda r: r.get("direction") in ("input", "output")),
    (
        "dtype",
        lambda r: _section_filled(r.get("llm_description", ""), "数据类型")
        or any(
            k in r.get("llm_description", "").lower()
            for k in ["float", "int", "bool", "dtype", "type"]
        )
        or _has_json_dtype(r),
    ),
    (
        "shape",
        lambda r: _section_filled(r.get("llm_description", ""), "维度约束")
        or any(
            k in r.get("llm_description", "").lower()
            for k in ["shape", "dim", "dimension", "维度", "形状"]
        ),
    ),
    (
        "optional",
        lambda r: _section_filled(r.get("llm_description", ""), "是否可选")
        or any(
            k in r.get("llm_description", "")
            for k in ["optional", "required", "mandatory", "必选", "可选"]
        ),
    ),
    (
        "discontinuous",
        # Only check for aclTensor parameters — non-tensor params (int64_t,
        # bool, const char*, etc.) legitimately have null discontinuous.
        # Using "acltensor" instead of "tensor" avoids false positives on
        # types like "int64_t" that don't contain "tensor".
        lambda r: "acltensor" in r.get("param_type", "").lower()
        and r.get("is_support_discontinuous") is not None,
    ),
]


def _context_has_attribute(context: str, attr_name: str) -> bool:
    """Check whether the original context contains keywords for *attr_name*."""
    keywords: dict[str, list[str]] = {
        "direction": ["输入", "输出", "input", "output"],
        "dtype": ["float", "int", "bool", "dtype", "type", "数据类型"],
        "shape": ["shape", "dim", "维度", "形状"],
        "optional": ["optional", "required", "必选", "可选"],
        "discontinuous": ["discontinuous", "连续", "non-contiguous"],
    }
    kws = keywords.get(attr_name, [])
    ctx_lower = context.lower()
    return any(kw.lower() in ctx_lower for kw in kws)


def attribute_checklist_verify(result: dict, context: str) -> dict:
    """Check which expected attributes are covered by the ``llm_description``.

    Only flags an attribute as *missing* if the original context actually
    contains keywords for it — avoids false positives when the document simply
    doesn't mention a property.
    """
    desc = result.get("llm_description", "")
    missing_attrs: list[str] = []
    for attr_name, check_fn in ATTR_CHECKLIST:
        if not check_fn(result):
            if _context_has_attribute(context, attr_name):
                missing_attrs.append(attr_name)

    if missing_attrs:
        logger.warning(
            "参数 %s: 缺失属性 %s（上下文中存在）",
            result.get("param_name"),
            missing_attrs,
        )

    return {
        "param_name": result.get("param_name"),
        "missing_attrs": missing_attrs,
        "desc_length": len(desc),
        "desc_too_short": len(desc) < 30,
    }


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def _build_coverage_report(
    all_results: list[dict],
    all_params: list[dict],
) -> dict:
    """Generate coverage statistics across all extracted parameters."""
    total = len(all_params)
    extracted = len(all_results)
    short_descs = sum(
        1 for r in all_results if len(r.get("llm_description", "")) < 30
    )
    with_missing = sum(
        1
        for r in all_results
        if r.get("_validation", {}).get("missing_attrs")
    )

    return {
        "total_params": total,
        "extracted": extracted,
        "not_extracted": total - extracted,
        "short_descriptions": short_descs,
        "with_missing_attrs": with_missing,
        "coverage_rate": f"{extracted / total * 100:.1f}%" if total else "N/A",
    }


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

async def validate_results_node(state: DescriptionExtractState) -> dict[str, Any]:
    """Validate extraction results and produce a coverage report."""
    ws_results = state.get("ws_results", [])
    exe_results = state.get("exe_results", [])
    all_results = ws_results + exe_results
    all_params = state.get("parameters", [])

    logger.info(
        "ValidateResults: %d ws + %d exe = %d total",
        len(ws_results),
        len(exe_results),
        len(all_results),
    )

    # Per-parameter verification
    validation_report: dict[str, dict] = {}
    for result in all_results:
        context = result.get("_context", "")
        report = attribute_checklist_verify(result, context)
        pname = result["param_name"]
        validation_report[pname] = report
        # Attach report back to the result dict for save_descriptions
        result["_validation"] = report

    # Coverage statistics
    coverage = _build_coverage_report(all_results, all_params)

    logger.info(
        "ValidateResults: coverage=%s, not_extracted=%d, with_missing_attrs=%d, short=%d",
        coverage["coverage_rate"],
        coverage["not_extracted"],
        coverage["with_missing_attrs"],
        coverage["short_descriptions"],
    )

    return {
        "validation_report": validation_report,
        "coverage_report": coverage,
        "error": None,
    }

