"""Determinism extract node: extract determinism info from constraints section via LLM."""

from __future__ import annotations

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import DETERMINISM_EXTRACT_PROMPT
from agent.utils.llm_common import create_llm, parse_json_response

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def determinism_extract_node(state: PipelineState) -> dict[str, Any]:
    """Extract determinism info from the constraints section via LLM.

    Reads constraints section via MCP, calls LLM to extract determinism records,
    expands empty product to all supported platforms, saves results via MCP.
    """
    doc_id = state.get("doc_id")
    operator_name = state.get("operator_name")
    if not doc_id or not operator_name:
        logger.warning("determinism_extract: missing doc_id or operator_name, skipping")
        return {"error": None}

    try:
        # Get constraints section
        section = await _mcp_client.get_section(doc_id, "constraints")
        if not section or not section.get("content"):
            logger.info("determinism_extract: no constraints section for doc_id=%s, skipping", doc_id)
            return {"error": None}

        section_content = section["content"]
        if "确定性" not in section_content:
            logger.info("determinism_extract: no determinism mention in constraints, skipping")
            return {"error": None}

        # Get supported platforms for expansion
        platforms_data = await _mcp_client.query_platform_support_by_operator(operator_name)
        supported_platforms = [
            p["platform_name"] for p in platforms_data if p.get("is_supported")
        ]
        platform_list = "\n".join(f"- {p}" for p in supported_platforms) if supported_platforms else "(无已知平台)"

        # Call LLM
        llm = create_llm()
        prompt = DETERMINISM_EXTRACT_PROMPT.format(
            platform_list=platform_list,
            section_content=section_content,
        )
        response = await llm.ainvoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)

        # Parse LLM response
        records = _parse_llm_response(raw_text)
        if not records:
            logger.info("determinism_extract: LLM returned no records for %s", operator_name)
            return {"error": None}

        # Expand empty product to all supported platforms
        expanded_records = _expand_platforms(records, supported_platforms)

        # Save via MCP
        result = await _mcp_client.save_determinism(doc_id, expanded_records)
        logger.info(
            "determinism_extract: saved %d records for %s (doc_id=%s)",
            result.get("saved", 0),
            operator_name,
            doc_id,
        )
        return {"determinism_records": expanded_records, "error": None}

    except Exception as e:
        logger.exception("determinism_extract failed for %s", operator_name)
        return {"error": str(e)}


def _parse_llm_response(raw_text: str) -> list[dict]:
    """Parse LLM JSON response, handling code blocks and bare JSON."""
    data = parse_json_response(raw_text, list)
    if not isinstance(data, list):
        return []
    return [
        {
            "product": item.get("product", ""),
            "value": bool(item.get("value", False)),
            "src_text": item.get("src_text", ""),
        }
        for item in data
        if isinstance(item, dict)
    ]


def _expand_platforms(records: list[dict], supported_platforms: list[str]) -> list[dict]:
    """Expand records with empty product to all supported platforms."""
    expanded = []
    for record in records:
        product = record.get("product", "").strip()
        if product:
            # Has specific platform, keep as-is
            expanded.append(record)
        else:
            # No platform specified, expand to all supported platforms
            for platform in supported_platforms:
                expanded.append({
                    "product": platform,
                    "value": record["value"],
                    "src_text": record["src_text"],
                })
    return expanded
