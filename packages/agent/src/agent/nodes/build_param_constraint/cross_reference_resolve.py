"""CrossReferenceResolve node: resolve cross-parameter references.

When a parameter's shape/dtype/dformat is empty or a cross-reference like
"与query一致", this node resolves it by copying the value from the
referenced parameter.

Detection covers 3 levels:
  1. Field value itself is a cross-reference (e.g. shape = "与query一致")
  2. usage_notes contains a cross-reference with attribute keyword
     (e.g. "输入维度需要与query、value保持一致")
  3. llm_description contains a cross-reference

Runs after fetch_param_data, before attrs_build/dimensions_agent.
Modifies params in-memory (state update) -- no DB writes needed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.nodes.build_param_constraint._helpers import _parse_json_field
from agent.nodes.build_param_constraint.state import BuildParamConstraintState
from agent.utils.param_validators import is_cross_reference

logger = logging.getLogger(__name__)

# Fields that can be cross-referenced, paired with their DB-alias fields
# so both stay consistent when a value is copied.
_REFERENCE_FIELDS: list[tuple[str, str | None]] = [
    ("shape", None),
    ("dtype_desc", "data_type"),
    ("dformat_desc", "data_format"),
]

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches "与query一致", "和query、value保持一致", "同query相同" etc.
# Captures one or more param names separated by 、 or , (Chinese/ASCII comma).
_CROSS_REF_RE = re.compile(
    r"[与和同跟]\s*"
    r"((?:[a-zA-Z_][a-zA-Z0-9_]*\s*[、,，]\s*)*[a-zA-Z_][a-zA-Z0-9_]*)"
    r"\s*(?:保持)?(?:一致|相同|一样|同)",
    re.UNICODE,
)

# Attribute keyword -> field name mapping.
# When a cross-reference appears in usage_notes / llm_description, nearby
# attribute keywords determine which field to resolve.
_ATTR_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["维度", "shape", "输入维度", "输出维度"], "shape"),
    (["数据类型", "dtype", "类型"], "dtype_desc"),
    (["数据格式", "format", "格式"], "dformat_desc"),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_target_params(text: str) -> list[str]:
    """Extract target parameter names from a cross-reference text.

    "与query、value保持一致" -> ["query", "value"]
    "和query相同"           -> ["query"]
    """
    targets: list[str] = []
    for m in _CROSS_REF_RE.finditer(text):
        raw = m.group(1)
        for part in re.split(r"[、,，]", raw):
            part = part.strip()
            if part:
                targets.append(part)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _detect_attr_type(text: str, ref_start: int) -> str | None:
    """Determine which field a cross-reference refers to.

    Looks for attribute keywords (维度/数据类型/数据格式) in the text
    preceding the cross-reference match.  Falls back to scanning the
    full text if no keyword is found in the prefix.
    """
    prefix = text[:ref_start]
    for keywords, field in _ATTR_KEYWORD_MAP:
        for kw in keywords:
            if kw in prefix:
                return field
    # Fallback: scan full text
    for keywords, field in _ATTR_KEYWORD_MAP:
        for kw in keywords:
            if kw in text:
                return field
    return None


def _field_is_empty(field_value: str) -> bool:
    """Check if a JSON field value is empty or has no real values."""
    if not field_value:
        return True
    parsed = _parse_json_field(field_value)
    if not parsed:
        return True
    return all(not v for v in parsed.values())


def _field_has_cross_ref(field_value: str) -> bool:
    """Check if any platform value in the field is a cross-reference."""
    if not field_value:
        return False
    parsed = _parse_json_field(field_value)
    for val in parsed.values():
        if val and is_cross_reference(val):
            return True
    return False


def _field_needs_resolution(field_value: str) -> bool:
    """True when the field is empty OR contains a cross-reference."""
    return _field_is_empty(field_value) or _field_has_cross_ref(field_value)


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


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _detect_cross_refs_for_param(
    param: dict,
) -> dict[str, list[str]]:
    """Detect cross-references for a single parameter.

    Returns ``{field_name: [target_param_names]}``.
    Only includes fields that *need* resolution (empty or cross-ref).
    """
    refs: dict[str, list[str]] = {}

    # -- Level 1: field value itself is a cross-reference --
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

    # -- Level 2 & 3: usage_notes / llm_description --
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

    # Deduplicate targets per field (preserve order)
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


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _copy_field(
    target_param: dict,
    source_param: dict,
    field: str,
    alias: str | None,
) -> bool:
    """Copy *field* (and its alias) from *source_param* to *target_param*.

    Returns True if the copy succeeded (source had a resolved value).
    """
    source_val = source_param.get(field, "")
    if not _field_is_resolved(source_val):
        return False
    target_param[field] = source_val
    if alias:
        target_param[alias] = source_val
    return True


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


async def cross_reference_resolve_node(
    state: BuildParamConstraintState,
) -> dict[str, Any]:
    """Resolve cross-parameter references in shape/dtype/dformat fields.

    Runs after fetch_param_data, before attrs_build/dimensions_agent.
    Modifies params in-memory (state update) -- no DB writes needed.
    """
    params = state.get("params", [])
    if not params:
        return {"params": []}

    # Shallow-copy each param so we can mutate field values
    updated: list[dict] = [dict(p) for p in params]
    index: dict[str, dict] = {p["param_name"]: p for p in updated}

    # Detect all cross-references
    all_refs: dict[str, dict[str, list[str]]] = {}
    for p in updated:
        pname = p["param_name"]
        refs = _detect_cross_refs_for_param(p)
        # Discard self-references
        for field in refs:
            refs[field] = [t for t in refs[field] if t != pname]
        if refs:
            all_refs[pname] = refs

    if not all_refs:
        logger.info("CrossReferenceResolve: no cross-references detected")
        return {"params": updated}

    # Iterative topological resolution.
    # Each pass: for params still needing resolution, try to copy from
    # target params that already have resolved values.  Repeat until no
    # progress (handles chains A->B->C and stops on circular refs).
    resolved: set[str] = set()
    for iteration in range(len(updated)):
        progress = False
        for pname, field_refs in all_refs.items():
            if pname in resolved:
                continue
            param = index[pname]
            for field, targets in field_refs.items():
                alias = dict(_REFERENCE_FIELDS).get(field)
                # Skip if field already has a real value
                if _field_is_resolved(param.get(field, "")):
                    resolved.add(pname)
                    progress = True
                    break
                # Try each target param in order
                for target in targets:
                    target_param = index.get(target)
                    if target_param is None:
                        continue
                    if _copy_field(param, target_param, field, alias):
                        logger.info(
                            "CrossReferenceResolve: %s.%s <- %s.%s",
                            pname, field, target, field,
                        )
                        resolved.add(pname)
                        progress = True
                        break
                if pname in resolved:
                    break

        if not progress:
            break  # circular refs or unresolved targets

    unresolved = set(all_refs.keys()) - resolved
    if unresolved:
        logger.warning(
            "CrossReferenceResolve: %d params could not be resolved: %s",
            len(unresolved),
            ", ".join(sorted(unresolved)),
        )

    logger.info(
        "CrossReferenceResolve: resolved %d/%d cross-referenced params",
        len(resolved),
        len(all_refs),
    )
    return {"params": updated}
