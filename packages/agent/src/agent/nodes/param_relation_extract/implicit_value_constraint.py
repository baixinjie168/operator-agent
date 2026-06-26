"""ImplicitValueConstraint node: extract value constraints for implicit params.

Implicit dimension variables (K1, K2, M, S, H, etc.) are extracted by
implicit_param_extract, but their VALUE constraints (e.g. "K1<65536",
"S<=1024", "H=32/64") are not extracted by any existing node.

This node scans the constraints section text for value constraints
on known implicit variables using deterministic regex patterns.

Position: inside param_relation_extract subgraph, after implicit_param_extract.
Zero LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.state import RelationExtractState

logger = logging.getLogger(__name__)
_mcp_client = MCPClient()

# Patterns: (regex, expr_template, expr_type)
# Each pattern uses {var} as placeholder for the variable name.
_VALUE_PATTERNS = [
    # "X<65536" / "X小于65536" / "X不超过65536"
    (
        re.compile(r"(\w+)\s*(?:<|小于|不超过|不大于)\s*(\d+)"),
        "{var} < {n}",
        "self_value_range",
    ),
    # "X<=1024" / "X最大为1024"
    (
        re.compile(r"(\w+)\s*(?:<=|最大为|最多为)\s*(\d+)"),
        "{var} <= {n}",
        "self_value_range",
    ),
    # "X>0" / "X大于0" / "X必须大于0"
    (
        re.compile(r"(\w+)\s*(?:>|大于|必须大于)\s*(\d+)"),
        "{var} > {n}",
        "self_value_range",
    ),
    # "X>=1" / "X最小为1"
    (
        re.compile(r"(\w+)\s*(?:>=|最小为|至少为)\s*(\d+)"),
        "{var} >= {n}",
        "self_value_range",
    ),
    # "X=32/64" / "X只支持32/64" / "X取值32、64"
    (
        re.compile(r"(\w+)\s*(?:=|为|取值|只支持|仅支持)\s*(\d+(?:\s*[/、，,]\s*\d+)+)"),
        "{var}.range_value in [{values}]",
        "self_value_enum",
    ),
    # "X为偶数" / "X必须是偶数"
    (
        re.compile(r"(\w+)\s*(?:必须)?是偶数"),
        "{var} % 2 == 0",
        "self_alignment",
    ),
]


def _extract_enum_values(raw: str) -> str:
    """Parse '32/64' or '32、64' into '32, 64'."""
    parts = re.split(r"[/、，,]", raw)
    return ", ".join(p.strip() for p in parts if p.strip().isdigit())


async def implicit_value_constraint_node(state: RelationExtractState) -> dict[str, Any]:
    """Extract value constraints for implicit dimension variables.

    Scans constraint section text for patterns like "K1<65536" and
    generates constraint expressions for known implicit variables.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")
    mappings = state.get("implicit_params", [])
    ws_content = state.get("ws_section_content", "")
    exe_content = state.get("exe_section_content", "")
    sections_text = f"{ws_content}\n\n{exe_content}"

    logger.info("ImplicitValueConstraint: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id or not mappings or not sections_text.strip():
        return {"implicit_value_constraints": [], "error": None}

    # Collect known implicit variable names (exclude constants/external)
    var_names = {
        m["var_name"] for m in mappings
        if m.get("var_name")
        and not m.get("is_constant")
        and not m.get("is_external_constant")
        and not m.get("is_quantization_type")
    }

    if not var_names:
        return {"implicit_value_constraints": [], "error": None}

    constraints: list[dict] = []

    for pattern, template, expr_type in _VALUE_PATTERNS:
        for m in pattern.finditer(sections_text):
            var = m.group(1)
            if var not in var_names:
                continue

            # Build expr from template
            if "{values}" in template:
                values_str = _extract_enum_values(m.group(2))
                expr = template.format(var=var, values=values_str)
            else:
                expr = template.format(var=var, n=m.group(2))

            # Get source context (±50 chars)
            start = max(0, m.start() - 50)
            end = min(len(sections_text), m.end() + 50)
            src_text = sections_text[start:end].strip()

            # Dedup
            if any(c["relation_object"]["expr"] == expr for c in constraints):
                continue

            constraints.append({
                "expr_type": expr_type,
                "expr": expr,
                "relation_params": [var],
                "src_text": src_text,
            })

    if constraints:
        # Persist as param_relations with relation_type=self_constraint
        existing = await _mcp_client.query_param_relations(doc_id)
        fn_name = ""  # implicit params are not function-scoped
        new_rels = []
        for c in constraints:
            new_rels.append({
                "function_name": fn_name,
                "relation_type": "self_constraint",
                "platform": "",
                "description": c["expr"],
                "params": c["relation_params"],
                "param_optional": {c["relation_params"][0]: False},
                "source_citation": c["src_text"],
                "relation_object": {
                    "expr_type": c["expr_type"],
                    "expr": c["expr"],
                    "relation_params": c["relation_params"],
                    "src_text": c["src_text"],
                },
            })
        merged = existing + new_rels
        await _mcp_client.save_param_relations(doc_id, merged)
        logger.info(
            "ImplicitValueConstraint: added %d value constraints for %s (vars=%s)",
            len(constraints), operator_name, sorted(var_names),
        )
    else:
        logger.info("ImplicitValueConstraint: no value constraints found for %s", operator_name)

    return {"implicit_value_constraints": constraints, "error": None}
