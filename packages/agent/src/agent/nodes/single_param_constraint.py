"""BuildSingleParamConstraint node: extract single-parameter constraints
and append them to the param_relations table.

Two-layer extraction:
- Layer 1: Deterministic regex matching (zero LLM cost, ~93% coverage)
- Layer 2: LLM extraction for long-tail patterns (~7%)

The node runs after BuildParamRelations so it can read existing multi-param
relations for deduplication. Results are appended to param_relations and
automatically flow into constraints_in_parameters via AssembleResult.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from agent.core.config import settings
from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_CONCURRENCY_LIMIT = 5


# ---------------------------------------------------------------------------
# Layer 1: Deterministic regex rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleParamRule:
    """A deterministic single-parameter constraint rule.

    Attributes:
        pattern: Regex pattern to match in parameter text.
        expr_template: Python expression template. Uses ``{param}`` for
            the parameter name and ``{n}`` for a captured numeric group.
        expr_type: The expr_type value stored in relation_object.
        description_template: Human-readable description template.
        param_type_filter: If set, only apply to params whose type
            contains this substring (e.g. ``"aclTensorList"``).
    """

    pattern: str
    expr_template: str
    expr_type: str
    description_template: str
    param_type_filter: str = ""

    _compiled: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def search(self, text: str) -> re.Match | None:
        return self._compiled.search(text)


RULES: list[SingleParamRule] = [
    # --- A. Empty Tensor ---
    SingleParamRule(
        pattern=r"不支持\s*空\s*Tensor|不允许\s*空\s*Tensor",
        expr_template="all(d > 0 for d in {param}.shape)",
        expr_type="self_shape_nonempty",
        description_template="{param} 不支持空Tensor，所有维度必须大于0",
    ),
    # --- B. TensorList internal consistency ---
    SingleParamRule(
        pattern=r"该参数中所有\s*Tensor\s*的数据类型保持一致",
        expr_template=(
            "all({param}[i].dtype == {param}[0].dtype"
            " for i in range(len({param})))"
        ),
        expr_type="self_dtype_consistency",
        description_template="{param} 中所有Tensor的数据类型必须保持一致",
        param_type_filter="aclTensorList",
    ),
    SingleParamRule(
        pattern=r"该参数中所有\s*Tensor\s*的数据格式保持一致",
        expr_template=(
            "all({param}[i].format == {param}[0].format"
            " for i in range(len({param})))"
        ),
        expr_type="self_format_consistency",
        description_template="{param} 中所有Tensor的数据格式必须保持一致",
        param_type_filter="aclTensorList",
    ),
    SingleParamRule(
        pattern=(
            r"该参数中所有\s*Tensor\s*的\s*shape\s*保持一致"
            r"|该参数中所有\s*Tensor\s*的维度保持一致"
        ),
        expr_template=(
            "all({param}[i].shape == {param}[0].shape"
            " for i in range(len({param})))"
        ),
        expr_type="self_shape_consistency",
        description_template="{param} 中所有Tensor的shape必须保持一致",
        param_type_filter="aclTensorList",
    ),
    # --- C. Shape upper bound ---
    SingleParamRule(
        pattern=(
            r"Tensor\s*维度超过\s*(\d+)\s*维"
            r"|维度超过\s*(\d+)\s*维"
            r"|shape\s*维度不高于\s*(\d+)\s*维"
            r"|维度不高于\s*(\d+)\s*维"
            r"|shape\s*支持\s*0[-~](\d+)\s*维"
        ),
        expr_template="len({param}.shape) <= {n}",
        expr_type="self_shape_upper_bound",
        description_template="{param} 的维度数不能超过{n}",
    ),
]


# ---------------------------------------------------------------------------
# Text collection helpers
# ---------------------------------------------------------------------------


def _get_param_text(param: dict) -> str:
    """Collect all available text for a parameter from DB fields."""
    parts: list[str] = []
    for field_name in ("param_desc", "llm_description", "src_content"):
        val = (param.get(field_name) or "").strip()
        if val:
            parts.append(val)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 1: rule matching
# ---------------------------------------------------------------------------


def _is_already_covered(
    param_name: str,
    rule: SingleParamRule,
    existing_relations: list[dict],
) -> bool:
    """Check whether an existing multi-param relation already expresses
    the same constraint for *param_name*.

    This prevents duplicate extraction when, e.g., a multi-param relation
    already contains ``all(d > 0 for d in x.shape)`` in its expr.
    """
    for r in existing_relations:
        if param_name not in r.get("params", []):
            continue
        obj = r.get("relation_object", {})
        expr = obj.get("expr", "")
        if not expr:
            continue

        et = rule.expr_type
        if et == "self_shape_nonempty":
            if "d > 0" in expr and param_name in expr:
                return True
        elif "consistency" in et:
            if f"{param_name}[0]" in expr:
                return True
        elif et == "self_shape_upper_bound":
            if f"len({param_name}.shape)" in expr:
                return True

    return False


def _extract_numeric_group(match: re.Match) -> str:
    """Return the first non-None numeric capture group from *match*."""
    for g in match.groups():
        if g and g.isdigit():
            return g
    return ""


def _match_rules(
    param: dict,
    existing_relations: list[dict],
) -> list[dict]:
    """Apply all deterministic rules to a single parameter.

    Returns a list of relation dicts for every matched (and not-yet-covered)
    rule.
    """
    combined = _get_param_text(param)
    if not combined.strip():
        return []

    pname: str = param["param_name"]
    ptype: str = param.get("param_type", "")
    fn: str = param.get("function_name", "")

    results: list[dict] = []

    for rule in RULES:
        # Type filter
        if rule.param_type_filter and rule.param_type_filter not in ptype:
            continue

        m = rule.search(combined)
        if m is None:
            continue

        # Dedup against existing multi-param relations
        if _is_already_covered(pname, rule, existing_relations):
            continue

        n_val = _extract_numeric_group(m)

        expr = rule.expr_template.format(param=pname, n=n_val)
        desc = rule.description_template.format(param=pname, n=n_val)
        src = m.group(0)

        results.append(
            {
                "function_name": fn,
                "relation_type": "self_constraint",
                "platform": "",
                "description": desc,
                "params": [pname],
                "param_optional": {pname: False},
                "source_citation": src,
                "relation_object": {
                    "expr_type": rule.expr_type,
                    "expr": expr,
                    "relation_params": [pname],
                    "src_text": src,
                },
                "_source": "deterministic",
            }
        )

    return results


# ---------------------------------------------------------------------------
# Layer 2: LLM long-tail extraction
# ---------------------------------------------------------------------------


async def _extract_long_tail(
    param: dict,
    existing_relations: list[dict],
    llm: ChatOpenAI,
    sem: asyncio.Semaphore,
) -> list[dict]:
    """Use LLM to extract non-pattern single-parameter constraints."""
    from agent.nodes.single_param_prompt import (
        SINGLE_PARAM_EXTRACT_PROMPT,
        parse_response,
    )

    combined = _get_param_text(param)
    if not combined.strip():
        return []

    pname: str = param["param_name"]
    ptype: str = param.get("param_type", "")
    fn: str = param.get("function_name", "")

    async with sem:
        try:
            prompt = SINGLE_PARAM_EXTRACT_PROMPT.format(
                param_name=pname,
                param_type=ptype,
                param_text=combined,
            )
            response = await llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            raw_results = parse_response(text)
        except Exception:
            logger.warning(
                "SingleParam: LLM extraction failed for param %s", pname,
            )
            return []

    # Post-process: stamp function_name, source tag, and build relation_object
    results: list[dict] = []
    for item in raw_results:
        expr_type = item.get("expr_type", "")
        expr = item.get("expr", "")
        desc = item.get("description", "")
        src = item.get("source_citation", "")
        params_list = item.get("params", [pname])

        results.append(
            {
                "function_name": fn,
                "relation_type": "self_constraint",
                "platform": item.get("platform", ""),
                "description": desc,
                "params": params_list,
                "param_optional": {pname: False},
                "source_citation": src,
                "relation_object": {
                    "expr_type": expr_type,
                    "expr": expr,
                    "relation_params": params_list,
                    "src_text": src,
                },
                "_source": "llm_long_tail",
            }
        )

    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _dedup(relations: list[dict]) -> list[dict]:
    """Deduplicate single-parameter constraints.

    Key: (function_name, relation_type, params_tuple, expr_type).
    Priority: ``deterministic`` source wins over ``llm_long_tail``.
    """
    seen: dict[tuple, dict] = {}
    for r in relations:
        obj = r.get("relation_object", {})
        key = (
            r.get("function_name", ""),
            r.get("relation_type", ""),
            tuple(r.get("params", [])),
            obj.get("expr_type", ""),
        )
        if key not in seen:
            seen[key] = r
        elif r.get("_source") == "deterministic":
            seen[key] = r  # deterministic always wins
    return list(seen.values())


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


def _log_coverage(
    params: list[dict],
    all_new: list[dict],
    layer1_count: int,
    layer2_count: int,
) -> None:
    """Log coverage statistics for single-parameter constraints."""
    covered_params: set[str] = set()
    for r in all_new:
        for p in r.get("params", []):
            covered_params.add(p)

    tensor_params = [
        p for p in params if "aclTensor" in p.get("param_type", "")
    ]
    uncovered = [
        p["param_name"]
        for p in tensor_params
        if p["param_name"] not in covered_params
    ]

    logger.info(
        "SingleParam coverage: %d/%d tensor params covered "
        "(L1=%d, L2=%d, uncovered=%s)",
        len(covered_params),
        len(tensor_params),
        layer1_count,
        layer2_count,
        uncovered[:10],
    )


# ---------------------------------------------------------------------------
# Main node entry point
# ---------------------------------------------------------------------------


async def build_single_param_constraint_node(
    state: PipelineState,
) -> dict[str, Any]:
    """Extract single-parameter constraints and append to param_relations.

    Flow:
    1. Load params + existing multi-param relations from DB.
    2. Layer 1: apply deterministic regex rules to every parameter.
    3. Layer 2 (optional): use LLM for long-tail patterns.
    4. Deduplicate and append to param_relations table.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info(
        "SingleParamConstraint: doc_id=%s for %s", doc_id, operator_name,
    )

    if not doc_id:
        logger.warning("SingleParamConstraint: no doc_id, skipping")
        return {"error": None}

    try:
        # Step 1: Load data
        params = await _mcp_client.query_params_by_doc_id(doc_id)
        existing = await _mcp_client.query_param_relations(doc_id)

        if not params:
            logger.info("SingleParamConstraint: no parameters, skipping")
            return {"error": None}

        # Step 2: Layer 1 — Deterministic regex
        layer1_results: list[dict] = []
        for param in params:
            layer1_results.extend(_match_rules(param, existing))

        logger.info(
            "SingleParamConstraint: Layer 1 matched %d constraints",
            len(layer1_results),
        )

        # Step 3: Layer 2 — LLM long-tail (optional, controlled by config)
        layer2_results: list[dict] = []
        if getattr(settings, "enable_single_param_llm", False):
            try:
                llm = ChatOpenAI(
                    api_key=settings.active_api_key,
                    base_url=settings.active_base_url,
                    model=settings.active_model,
                    temperature=0.1,
                )
            except Exception:
                logger.exception(
                    "SingleParamConstraint: failed to create LLM, "
                    "skipping Layer 2",
                )
                llm = None

            if llm is not None:
                sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)
                tasks = [
                    _extract_long_tail(p, existing, llm, sem)
                    for p in params
                ]
                gathered = await asyncio.gather(*tasks)
                for rels in gathered:
                    layer2_results.extend(rels)

                logger.info(
                    "SingleParamConstraint: Layer 2 extracted %d constraints",
                    len(layer2_results),
                )

        # Step 4: Merge and deduplicate
        all_new = _dedup(layer1_results + layer2_results)

        # Step 5: Append to param_relations
        if all_new:
            merged = existing + all_new
            result = await _mcp_client.save_param_relations(doc_id, merged)
            logger.info(
                "SingleParamConstraint: saved %d total relations "
                "(%d existing + %d new)",
                result.get("saved", 0),
                len(existing),
                len(all_new),
            )
        else:
            logger.info("SingleParamConstraint: no new constraints found")

        # Step 6: Coverage report
        _log_coverage(params, all_new, len(layer1_results), len(layer2_results))

        return {
            "single_param_constraints": all_new,
            "error": None,
        }

    except Exception:
        logger.exception(
            "SingleParamConstraint failed for %s", operator_name,
        )
        return {"error": "single_param_constraint_failed"}

