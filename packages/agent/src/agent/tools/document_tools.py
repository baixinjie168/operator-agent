"""Document processing tools — thin wrappers around MCP server calls."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from agent.mcp_client import MCPClient

logger = logging.getLogger(__name__)

_client = MCPClient()


@tool
async def parse_document(content: str) -> dict[str, Any]:
    """Parse a CANN operator Markdown document into structured sections.

    Splits the document by headings, classifies each section, extracts product
    support tables and function signatures. Returns a dict matching
    ParsedOperatorDocument schema.

    Args:
        content: Raw Markdown content of the operator document.
    """
    return await _client.parse_doc(content)


@tool
async def check_document_version(operator_name: str, content_hash: str) -> dict[str, Any]:
    """Check if a document has changed from the previously stored version.

    Args:
        operator_name: Operator name extracted from the H1 title.
        content_hash: SHA256 hex digest of the document content.
    """
    return await _client.check_version(operator_name, content_hash)


@tool
async def save_document(operator_name: str, content: str) -> dict[str, Any]:
    """Save a raw document to the database.

    Args:
        operator_name: Operator name.
        content: Raw Markdown content.
    """
    return await _client.save_doc(operator_name, content)


@tool
async def save_parsed_result(
    operator_name: str,
    version: int,
    parsed_data: str,
) -> str:
    """Save parsed document data to the database.

    Args:
        operator_name: Operator name.
        version: Document version number.
        parsed_data: JSON string of the parsed document data.
    """
    return await _client.save_parsed(
        operator_name,
        version,
        json.loads(parsed_data),
    )


@tool
async def get_parsed_document(
    operator_name: str,
    version: int | None = None,
) -> dict[str, Any] | None:
    """Retrieve previously parsed document data from the database.

    Args:
        operator_name: Operator name.
        version: Optional version number. Returns the latest if omitted.
    """
    return await _client.get_parsed(operator_name, version)
