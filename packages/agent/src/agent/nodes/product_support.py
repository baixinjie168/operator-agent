"""ProductSupport node: extracts product support info via LLM and persists to DB."""

import logging
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState
from agent.prompts import PRODUCT_SUPPORT_EXTRACT_PROMPT
from agent.core.llm import create_llm
from agent.utils.llm_common import parse_json_response

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()


async def product_support_node(state: PipelineState) -> dict[str, Any]:
    """Extract product support info via LLM and save to DB.

    Flow:
    1. Query parsed_data by doc_id via MCP
    2. Find section_type == "product_support" and get its content
    3. Call LLM to extract [{product, support}]
    4. Save to document_versions.product_support via MCP
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ProductSupport: received state doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ProductSupport: no doc_id in state, skipping")
        return {"product_support": [], "error": None}

    try:
        parsed = await _mcp_client.get_parsed_by_doc_id(doc_id)
        if not parsed:
            logger.warning("ProductSupport: no parsed_data for doc_id=%s", doc_id)
            return {"product_support": [], "error": None}

        content = _find_product_support_content(parsed.get("sections", []))
        if not content:
            logger.warning("ProductSupport: no product_support section for doc_id=%s", doc_id)
            return {"product_support": [], "error": None}

        products = await _extract_via_llm(content)
        logger.info(
            "ProductSupport: extracted %d products for %s",
            len(products),
            operator_name,
        )

        await _mcp_client.save_product_support(doc_id, products)
        logger.info("ProductSupport: saved %d products to document_versions for doc_id=%s", len(products), doc_id)

        # 同时保存到 platform_support 表
        platforms = [
            {
                "platform_name": p.get("product", ""),
                "is_supported": 1 if p.get("support", False) else 0,
            }
            for p in products
        ]
        await _mcp_client.save_platform_support(doc_id, platforms)
        logger.info("ProductSupport: saved %d platforms to platform_support for doc_id=%s", len(platforms), doc_id)

        return {"product_support": products, "error": None}

    except Exception as e:
        logger.exception("ProductSupport failed for %s", operator_name)
        return {"product_support": [], "error": str(e)}


def _find_product_support_content(sections: list[dict]) -> str | None:
    """Find the product_support section and return its content."""
    for section in sections:
        if section.get("section_type") == "product_support":
            return section.get("content")
    return None


async def _extract_via_llm(content: str) -> list[dict]:
    """Call LLM to extract product support info from markdown table content."""
    llm = create_llm()

    prompt = PRODUCT_SUPPORT_EXTRACT_PROMPT.format(content=content)
    response = await llm.ainvoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    return parse_json_response(text, list) or []
