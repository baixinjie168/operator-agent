"""MCP stdio client wrapper for the agent main system.

Manages the MCP server subprocess lifecycle and provides
async methods to call MCP tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.core.config import settings

logger = logging.getLogger(__name__)


class MCPClient:
    """Async client that communicates with the MCP server via stdio."""

    def __init__(self, server_command: str = "") -> None:
        if not server_command:
            server_command = settings.mcp_server_command
        parts = server_command.split()
        self._params = StdioServerParameters(command=parts[0], args=parts[1:])

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Start MCP server subprocess, call a tool, and return the result."""
        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

                if result.isError:
                    error_msg = result.content[0].text if result.content else "Unknown MCP error"
                    raise RuntimeError(f"MCP tool '{tool_name}' error: {error_msg}")

                if result.content:
                    text = result.content[0].text
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        return text
                return None

    async def check_version(self, operator_name: str, content_hash: str) -> dict:
        """Check document version status."""
        return await self._call_tool("check_version", {
            "operator_name": operator_name,
            "content_hash": content_hash,
        })

    async def save_doc(self, operator_name: str, content: str, source_url: str | None = None) -> dict:
        """Save a document to the database."""
        return await self._call_tool("save_doc", {
            "operator_name": operator_name,
            "content": content,
            "source_url": source_url,
        })

    async def parse_doc(self, content: str) -> dict:
        """Parse a document via MCP."""
        return await self._call_tool("parse_doc", {"content": content})

    async def get_parsed(self, operator_name: str, version: int | None = None) -> dict | None:
        """Retrieve previously parsed document."""
        return await self._call_tool("get_parsed", {
            "operator_name": operator_name,
            "version": version,
        })

    async def list_operators(self) -> list[dict]:
        """List all registered operators."""
        return await self._call_tool("query_operators", {})

    async def save_parsed(self, operator_name: str, version: int, parsed_data: dict) -> str:
        """Save parsed data to the database."""
        return await self._call_tool("save_parsed", {
            "operator_name": operator_name,
            "version": version,
            "parsed_data": json.dumps(parsed_data, ensure_ascii=False),
        })

    async def get_parsed_by_doc_id(self, doc_id: int) -> dict | None:
        """Retrieve parsed document data by document_versions primary key."""
        return await self._call_tool("get_parsed_by_doc_id", {"doc_id": doc_id})

    async def save_parameters(self, doc_id: int, parameters: list[dict]) -> dict:
        """Save parsed parameters for a specific document version."""
        return await self._call_tool("save_params", {
            "doc_id": doc_id,
            "parameters": json.dumps(parameters, ensure_ascii=False),
        })

    async def save_product_support(self, doc_id: int, product_support_data: list[dict]) -> str:
        """Save product support data for a specific document version."""
        return await self._call_tool("save_product_support", {
            "doc_id": doc_id,
            "product_support_data": json.dumps(product_support_data, ensure_ascii=False),
        })

    async def query_parameters(self, operator_name: str | None = None) -> list[dict]:
        """Query parameters, optionally filtered by operator name."""
        args: dict[str, Any] = {}
        if operator_name is not None:
            args["operator_name"] = operator_name
        return await self._call_tool("query_params", args)

    async def query_params_by_doc_id(self, doc_id: int) -> list[dict]:
        """Query parameters for a specific document version."""
        return await self._call_tool("query_params_by_doc", {"doc_id": doc_id})

    async def update_param_descriptions(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update parameter description fields."""
        return await self._call_tool("update_param_descs", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })
