"""Deterministic complexity classification for parameter relations.

Zero LLM cost — pure regex-based routing that determines whether a
relation should go through the DeepAgent (complex) or the existing
single-shot LLM path (simple).

Used by:
- build_param_relations._process_one  (type 4 routing)
- build_single_param_constraint_node   (type 3 routing)
"""

from __future__ import annotations

import re

# Keywords that indicate conditional / enum / presence logic
_COMPLEXITY_MARKERS = re.compile(
    r"当|如果|若|条件下|per-channel|per-tensor|per-group|"
    r"不为空|为空|is not None|存在|可选|"
    r"\[.*?\]\s*[，,/、]\s*\[.*?\]"   # multi-shape candidates [E,N1]/[N1]
)

# Implicit enum params that always require Agent handling
_IMPLICIT_ENUM_PARAMS = frozenset({"quantization_type"})


def is_complex_relation(rel: dict) -> bool:
    """Determine if a relation needs the DeepAgent (type 4).

    Criteria (any one triggers Agent routing):
    1. Conditional/enum/presence keywords + 3+ params
    2. Multi-shape candidate pattern ([A,B]/[C])
    3. Involves quantization_type or other implicit enum params
    """
    desc = rel.get("description", "")
    src = rel.get("source_citation", "")
    params = rel.get("params", [])
    combined = desc + " " + src

    # Rule 1: conditional keywords + 3+ params
    if _COMPLEXITY_MARKERS.search(combined) and len(params) >= 3:
        return True

    # Rule 2: multi-shape candidate pattern
    if re.search(r"\[.*?\]\s*[，,/、]\s*\[.*?\]", combined):
        return True

    # Rule 3: involves implicit enum params
    if any(p in _IMPLICIT_ENUM_PARAMS for p in params):
        return True

    return False


def is_self_constraint(rel: dict) -> bool:
    """Determine if a relation is a single-parameter self-constraint (type 3).

    These are handled by the Agent with self_constraint.md skill.
    """
    return (
        rel.get("relation_type") == "self_constraint"
        and len(rel.get("params", [])) <= 1
    )
