"""Shared utility functions for build_param_constraint sub-graph."""

from __future__ import annotations

import json
import re


def _parse_json_field(value: str | dict | None) -> dict:
    """Parse a JSON field from DB, tolerating legacy flat text.

    Returns a dict: {"platform": "value"} or {"*": "value"}.
    If value is already a dict, returns as-is.
    If value is legacy flat text, wraps as {"*": text}.
    """
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            return {"*": value}
        except (json.JSONDecodeError, TypeError):
            return {"*": value}
    return {}


def _split_csv(value: str) -> list[str]:
    """Split a comma, Chinese comma, or slash separated list into sorted unique items."""
    if not value:
        return []
    return sorted({v.strip() for v in re.split(r"[、，,/]", value) if v.strip()})


def _normalize_type(ptype: str) -> str:
    """Strip const, pointer *, and reference & from C type names."""
    ptype = re.sub(r'\bconst\b', '', ptype)
    ptype = ptype.replace('*', '').replace('&', '').strip()
    return ptype


# ---------------------------------------------------------------------------
# Enum extraction from usage_notes text
# ---------------------------------------------------------------------------
_ID = r'[A-Za-z_][A-Za-z0-9_]*'
_SEP = r'\s*[/、]\s*'
_ENUM_LIST_RE = re.compile(
    r'(' + _ID + r'(?:' + _SEP + _ID + r'){2,})'
)


def _extract_enum_from_text(text: str) -> list[str]:
    """Extract enumeration values from usage notes text.

    Detects patterns like "fastgelu/gelu/relu/silu".
    Returns sorted list of enum values, or [] if no enum pattern found.
    """
    if not text:
        return []
    matches = _ENUM_LIST_RE.findall(text)
    if not matches:
        return []
    all_values: set[str] = set()
    for match in matches:
        tokens = re.split(r'\s*[/、]\s*', match)
        for t in tokens:
            t = t.strip()
            if t and len(t) >= 2:
                all_values.add(t)
    return sorted(all_values)
