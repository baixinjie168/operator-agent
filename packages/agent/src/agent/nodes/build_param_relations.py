"""BuildParamRelations node: enrich param_relations with expr_type/expr and group by platform."""

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
from agent.prompts import RELATION_OBJECT_BUILD_PROMPT

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_CONCURRENCY_LIMIT = 5


def _format_signatures(sigs: list[dict]) -> str:
    """Build a concise signature text for LLM context."""
    if not sigs:
        return "（无函数签名信息）"
    lines: list[str] = []
    for sig in sigs:
        fn = sig.get("function_name", "")
        params = sig.get("parameters", [])
        param_strs = [f"{p.get('name', '')}: {p.get('type', '')}" for p in params]
        lines.append(f"{fn}({', '.join(param_strs)})")
    return "\n".join(lines)


def _parse_relation_object_response(text: str) -> dict[str, str]:
    """Parse LLM response into {expr_type, expr} dict."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "expr_type": data.get("expr_type", ""),
                "expr": data.get("expr", ""),
            }
    except json.JSONDecodeError:
        pass

    # Regex fallback
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict):
                return {
                    "expr_type": data.get("expr_type", ""),
                    "expr": data.get("expr", ""),
                }
        except json.JSONDecodeError:
            pass

    logger.warning("BuildParamRelations: failed to parse LLM response: %s", text[:200])
    return {"expr_type": "", "expr": ""}


async def _batch_extract_relation_objects(
    relations: list[dict],
    signatures_text: str,
) -> list[dict[str, str]]:
    """Batch LLM extraction of expr_type + expr for all relations.

    Returns a list of {expr_type, expr} dicts, one per relation.
    """
    if not relations:
        return []

    sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

    try:
        llm = ChatOpenAI(
            api_key=settings.active_api_key,
            base_url=settings.active_base_url,
            model=settings.active_model,
            temperature=0.1,
        )
    except Exception:
        logger.exception("BuildParamRelations: failed to create LLM")
        return [{"expr_type": "", "expr": ""}] * len(relations)

    async def _extract_one(rel: dict) -> dict[str, str]:
        async with sem:
            try:
                prompt = RELATION_OBJECT_BUILD_PROMPT.format(
                    signatures_text=signatures_text,
                    relation_type=rel.get("relation_type", ""),
                    params=json.dumps(rel.get("params", []), ensure_ascii=False),
                    description=rel.get("description", ""),
                    source_citation=rel.get("source_citation", ""),
                )
                response = await llm.ainvoke(prompt)
                text = response.content if hasattr(response, "content") else str(response)
                return _parse_relation_object_response(text)
            except Exception:
                logger.warning(
                    "BuildParamRelations: LLM failed for relation id=%s",
                    rel.get("id", "?"),
                )
                return {"expr_type": "", "expr": ""}

    results = await asyncio.gather(*[_extract_one(r) for r in relations])
    return list(results)


async def build_param_relations_node(state: PipelineState) -> dict[str, Any]:
    """Build relation_object for each param_relation row and group by platform.

    Flow:
    1. Query param_relations, function_signatures, platform_support
    2. LLM batch extract expr_type + expr from description
    3. Assemble relation_object per row and persist to DB
    4. Group by platform (precondition="无" → all supported platforms)
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("BuildParamRelations: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("BuildParamRelations: no doc_id, skipping")
        return {"error": None}

    try:
        # Step 1: Query data sources
        relations = await _mcp_client.query_param_relations(doc_id)
        sigs = await _mcp_client.query_function_signatures_by_doc_id(doc_id)
        platforms = await _mcp_client.query_platform_support_by_doc_id(doc_id)

        if not relations:
            logger.info("BuildParamRelations: no relations, skipping")
            return {"error": None}

        # Step 2: Build signature context
        signatures_text = _format_signatures(sigs)

        # Step 3: LLM batch extract expr_type + expr
        llm_results = await _batch_extract_relation_objects(relations, signatures_text)

        # Step 4: Assemble relation_object and persist
        updates: list[dict] = []
        for rel, llm_out in zip(relations, llm_results):
            relation_object = {
                "expr_type": llm_out.get("expr_type", ""),
                "expr": llm_out.get("expr", ""),
                "relation_params": rel.get("params", []),
                "src_text": rel.get("source_citation", ""),
            }
            updates.append({
                "id": rel["id"],
                "relation_object": json.dumps(relation_object, ensure_ascii=False),
            })

        result = await _mcp_client.update_param_relation_objects(doc_id, updates)
        logger.info(
            "BuildParamRelations: updated %d/%d relations (doc_id=%s)",
            result.get("updated", 0), len(updates), doc_id,
        )

        # Step 5: Group by platform
        supported_platforms = [
            p["platform_name"] for p in platforms if p.get("is_supported") == 1
        ]

        grouped: dict[str, list[dict]] = {}
        for rel, upd in zip(relations, updates):
            obj = json.loads(upd["relation_object"])
            precondition = rel.get("precondition", "无")
            if precondition == "无":
                targets = supported_platforms
            else:
                targets = [precondition]
            for plat in targets:
                grouped.setdefault(plat, []).append(obj)

        logger.info(
            "BuildParamRelations: grouped into %d platforms (doc_id=%s)",
            len(grouped), doc_id,
        )

        return {"error": None}

    except Exception as e:
        logger.exception("BuildParamRelations failed for %s", operator_name)
        return {"error": str(e)}
