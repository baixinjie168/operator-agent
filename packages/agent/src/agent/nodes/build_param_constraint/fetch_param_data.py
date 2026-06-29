"""FetchParamData node: one-shot DB queries + shared index building + value copy.

Value copy: reads param_relations for consistency constraints (e.g.
``A.shape == B.shape``) and copies actual shape/dtype values from source
params to params whose fields contain cross-reference text like "与query一致".
This replaces the old cross_reference_resolve node.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.build_param_constraint._helpers import _parse_json_field
from agent.nodes.build_param_constraint.state import BuildParamConstraintState
from agent.nodes.state import PipelineState
from agent.runtime.context import get_context
from agent.runtime.events import EventType, Span, SpanType
from agent.utils.param_validators import is_cross_reference

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

# Fields that can be cross-referenced, paired with their DB-alias fields
_REFERENCE_FIELDS: list[tuple[str, str | None]] = [
    ("shape", None),
    ("dtype_desc", "data_type"),
    ("dformat_desc", "data_format"),
]

# Matches "与query一致", "和query、value保持一致" etc.
_CROSS_REF_RE = re.compile(
    r"[与和同跟]\s*"
    r"((?:[a-zA-Z_][a-zA-Z0-9_]*\s*[、,，]\s*)*[a-zA-Z_][a-zA-Z0-9_]*)"
    r"\s*(?:保持)?(?:一致|相同|一样|同)",
    re.UNICODE,
)

_ATTR_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["维度", "shape", "输入维度", "输出维度"], "shape"),
    (["数据类型", "dtype", "类型"], "dtype_desc"),
    (["数据格式", "format", "格式"], "dformat_desc"),
]


def _field_is_resolved(field_value: str) -> bool:
    """True when the field has at least one real (non-cross-ref) value."""
    if not field_value:
        return False
    parsed = _parse_json_field(field_value)
    if not parsed:
        return False
    return any(
        val and not is_cross_reference(val)
        for val in parsed.values()
    )


def _field_needs_resolution(field_value: str) -> bool:
    """True when the field is empty OR contains a cross-reference."""
    if not field_value:
        return True
    parsed = _parse_json_field(field_value)
    if not parsed:
        return True
    return all(not v for v in parsed.values()) or any(
        v and is_cross_reference(v) for v in parsed.values()
    )


def _detect_attr_type(text: str, ref_start: int) -> str | None:
    """Determine which field a cross-reference refers to."""
    prefix = text[:ref_start]
    for keywords, field in _ATTR_KEYWORD_MAP:
        for kw in keywords:
            if kw in prefix:
                return field
    for keywords, field in _ATTR_KEYWORD_MAP:
        for kw in keywords:
            if kw in text:
                return field
    return None


def _extract_target_params(text: str) -> list[str]:
    """Extract target parameter names from a cross-reference text."""
    targets: list[str] = []
    for m in _CROSS_REF_RE.finditer(text):
        raw = m.group(1)
        for part in re.split(r"[、,，]", raw):
            part = part.strip()
            if part:
                targets.append(part)
    seen: set[str] = set()
    result: list[str] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _detect_cross_refs_for_param(param: dict) -> dict[str, list[str]]:
    """Detect cross-references for a single parameter.

    Returns {field_name: [target_param_names]}.
    Only includes fields that *need* resolution (empty or cross-ref).
    """
    refs: dict[str, list[str]] = {}

    # Level 1: field value itself is a cross-reference
    for field, _alias in _REFERENCE_FIELDS:
        val = param.get(field, "")
        if not val:
            continue
        parsed = _parse_json_field(val)
        for platform_val in parsed.values():
            if platform_val and is_cross_reference(platform_val):
                targets = _extract_target_params(platform_val)
                if targets:
                    refs.setdefault(field, []).extend(targets)

    # Level 2 & 3: usage_notes / llm_description
    for source_field in ("usage_notes", "llm_description"):
        source_val = param.get(source_field, "")
        if not source_val:
            continue
        parsed = _parse_json_field(source_val)
        texts = parsed.values() if parsed else [source_val]
        for text in texts:
            if not text:
                continue
            for m in _CROSS_REF_RE.finditer(text):
                targets = _extract_target_params(m.group(0))
                if not targets:
                    continue
                attr_type = _detect_attr_type(text, m.start())
                if not attr_type:
                    continue
                field_val = param.get(attr_type, "")
                if _field_needs_resolution(field_val):
                    refs.setdefault(attr_type, []).extend(targets)

    # Deduplicate targets per field
    deduped: dict[str, list[str]] = {}
    for field, targets in refs.items():
        seen: set[str] = set()
        unique: list[str] = []
        for t in targets:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        deduped[field] = unique

    return deduped


def _copy_field(
    target_param: dict,
    source_param: dict,
    field: str,
    alias: str | None,
) -> bool:
    """Copy *field* (and its alias) from *source_param* to *target_param*."""
    source_val = source_param.get(field, "")
    if not _field_is_resolved(source_val):
        return False
    target_param[field] = source_val
    if alias:
        target_param[alias] = source_val
    return True


def _resolve_cross_references(params: list[dict], relations: list[dict]) -> list[dict]:
    """Resolve cross-parameter references using consistency constraints.

    Uses two complementary detection mechanisms:
    1. Text-based: detect cross-reference text ("与self一致") in field values,
       usage_notes, or llm_description.
    2. Constraint-based: use type_equality / shape_equality constraints from
       param_relations (e.g. "A.dtype == B.dtype") to copy values even when
       no cross-reference text is present (e.g. dtype_desc is empty because
       table_column_extract skipped a relative-ref cell).

    Iteratively copies actual values from source params to params whose
    shape/dtype/dformat fields are empty or contain cross-reference text.
    """
    # Map expr attribute names to DB field names
    _EXPR_ATTR_TO_FIELD = {
        "shape": "shape",
        "dtype": "dtype_desc",
        "format": "dformat_desc",
        "dformat": "dformat_desc",
    }

    # Build attribute-specific consistency pairs from param_relations.
    # e.g. {"dtype_desc": {"gradOutput": {"self"}, "self": {"gradOutput"}}}
    attr_consistency_pairs: dict[str, dict[str, set[str]]] = {}
    for rel in relations:
        obj = rel.get("relation_object", {})
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(obj, dict):
            continue
        expr = obj.get("expr", "")
        if not expr:
            continue
        # Parse "A.shape == B.shape", "A.dtype == B.dtype" etc.
        m = re.match(r"([A-Za-z_]\w*)\.(\w+)\s*==\s*([A-Za-z_]\w*)\.(\w+)", expr)
        if m:
            p1, attr1, p2, attr2 = m.groups()
            if attr1 == attr2:
                field = _EXPR_ATTR_TO_FIELD.get(attr1)
                if field:
                    attr_consistency_pairs.setdefault(field, {})
                    attr_consistency_pairs[field].setdefault(p1, set()).add(p2)
                    attr_consistency_pairs[field].setdefault(p2, set()).add(p1)

    # Shallow-copy each param so we can mutate field values
    updated: list[dict] = [dict(p) for p in params]
    index: dict[str, dict] = {p["param_name"]: p for p in updated if p.get("param_name")}

    # Detect all cross-references
    all_refs: dict[str, dict[str, list[str]]] = {}
    for p in updated:
        pname = p.get("param_name", "")
        if not pname:
            continue
        refs = _detect_cross_refs_for_param(p)
        for field in refs:
            refs[field] = [t for t in refs[field] if t != pname]
        if refs:
            all_refs[pname] = refs

    if not all_refs and not attr_consistency_pairs:
        return updated

    # Iterative topological resolution (text-based)
    resolved: set[str] = set()
    if all_refs:
        for _iteration in range(len(updated)):
            progress = False
            for pname, field_refs in all_refs.items():
                if pname in resolved:
                    continue
                param = index[pname]
                for field, targets in field_refs.items():
                    alias = dict(_REFERENCE_FIELDS).get(field)
                    if _field_is_resolved(param.get(field, "")):
                        resolved.add(pname)
                        progress = True
                        break
                    for target in targets:
                        target_param = index.get(target)
                        if target_param is None:
                            continue
                        if _copy_field(param, target_param, field, alias):
                            logger.info(
                                "FetchParamData: %s.%s <- %s.%s (value copy)",
                                pname, field, target, field,
                            )
                            resolved.add(pname)
                            progress = True
                            break
                    if pname in resolved:
                        break

            if not progress:
                break

    # Constraint-based resolution pass: use type_equality / shape_equality
    # constraints from param_relations to resolve params that text-based
    # detection missed (e.g. dtype_desc is empty because the table cell
    # contained a relative-ref that was skipped by table_column_extract).
    constraint_resolved: set[str] = set()
    if attr_consistency_pairs:
        for _iteration in range(len(updated)):
            progress = False
            for field, pairs in attr_consistency_pairs.items():
                alias = dict(_REFERENCE_FIELDS).get(field)
                for pname, targets in pairs.items():
                    param = index.get(pname)
                    if param is None:
                        continue
                    # Skip if field already resolved
                    if _field_is_resolved(param.get(field, "")):
                        constraint_resolved.add(pname)
                        continue
                    for target in targets:
                        if target == pname:
                            continue
                        target_param = index.get(target)
                        if target_param is None:
                            continue
                        if _copy_field(param, target_param, field, alias):
                            logger.info(
                                "FetchParamData: %s.%s <- %s.%s (constraint-based)",
                                pname, field, target, field,
                            )
                            constraint_resolved.add(pname)
                            progress = True
                            break
            if not progress:
                break

    resolved.update(constraint_resolved)

    all_ref_params = set(all_refs.keys()) | set(
        pname
        for pairs in attr_consistency_pairs.values()
        for pname in pairs
    )
    unresolved = all_ref_params - resolved
    if unresolved:
        logger.warning(
            "FetchParamData: %d params could not be resolved: %s",
            len(unresolved),
            ", ".join(sorted(unresolved)),
        )

    logger.info(
        "FetchParamData: resolved %d/%d cross-referenced params",
        len(resolved),
        len(all_ref_params),
    )
    return updated


async def fetch_param_data_node(state: BuildParamConstraintState) -> dict[str, Any]:
    """Query all data sources, build shared indexes, and resolve cross-refs.

    Performs DB queries, then resolves cross-parameter references by copying
    actual values from source params (replaces the old cross_reference_resolve
    node).  The resolved params are passed to downstream nodes.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("FetchParamData: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("FetchParamData: no doc_id, skipping")
        return {"params": [], "error": None}

    try:
        # DB queries
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        sigs = await _mcp_client.query_function_signatures_by_doc_id(doc_id)
        platforms = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        dtype_combos = await _mcp_client.query_dtype_combos_by_doc_id(doc_id)
        relations = await _mcp_client.query_param_relations(doc_id)
        constraints_section = await _mcp_client.get_section(doc_id, "constraints")

        if not params:
            logger.info("FetchParamData: no parameters for doc_id=%s", doc_id)
            return {"params": [], "error": None}

        # Resolve cross-references (value copy from consistency constraints)
        params = _resolve_cross_references(params, relations)

        # Build indexes
        sig_type_map: dict[str, str] = {}
        all_sig_param_names: set[str] = set()
        for sig in sigs:
            for p in sig.get("parameters", []):
                key = f"{sig['function_name']}::{p['name']}"
                sig_type_map[key] = p.get("type", "")
                all_sig_param_names.add(p["name"])

        dtype_by_platform: dict[str, dict[str, set[str]]] = {}
        for combo in dtype_combos:
            plat = combo.get("platform", "common")
            dtype_by_platform.setdefault(plat, {})
            for pname, dtype_val in combo.get("combo", {}).items():
                dtype_by_platform[plat].setdefault(pname, set())
                if isinstance(dtype_val, str) and "/" in dtype_val:
                    for d in dtype_val.split("/"):
                        dtype_by_platform[plat][pname].add(d.strip())
                else:
                    dtype_by_platform[plat][pname].add(str(dtype_val))

        dtype_by_platform_lists: dict[str, dict[str, list[str]]] = {}
        for plat, params_map in dtype_by_platform.items():
            dtype_by_platform_lists[plat] = {
                pn: sorted(dtypes) for pn, dtypes in params_map.items()
            }

        supported_platforms = [
            p["platform_name"] for p in platforms if p.get("is_supported") == 1
        ]

        constraints_text = (constraints_section or {}).get("content", "") or ""

        logger.info(
            "FetchParamData: %d params, %d sigs, %d platforms, %d combos (doc_id=%s)",
            len(params), len(sigs), len(supported_platforms),
            len(dtype_combos), doc_id,
        )

        # NODE_PROGRESS
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
                "message": f"已加载 {len(params)} 个参数, {len(sigs)} 个签名, {len(dtype_combos)} 个 dtype 组合",
                "phase": "data_ready",
                "params_count": len(params),
                "sigs_count": len(sigs),
                "dtype_combos_count": len(dtype_combos),
            })

        return {
            "params": params,
            "sig_type_map": sig_type_map,
            "all_sig_param_names": sorted(all_sig_param_names),
            "dtype_by_platform": dtype_by_platform_lists,
            "supported_platforms": supported_platforms,
            "constraints_text": constraints_text,
            "param_relations": relations,
        }

    except Exception as e:
        logger.exception("FetchParamData failed for %s", operator_name)
        return {"params": [], "error": str(e)}
