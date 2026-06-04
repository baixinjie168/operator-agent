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
        async with stdio_client(self._params) as (read, write), ClientSession(read, write) as session:
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

    async def get_section(self, doc_id: int, section_type: str) -> dict | None:
        """Retrieve a specific section from parsed document by section_type."""
        return await self._call_tool("get_section", {
            "doc_id": doc_id,
            "section_type": section_type,
        })

    async def update_param_descriptions(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update parameter description fields."""
        return await self._call_tool("update_param_descs", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_shape(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update only the shape field of parameters."""
        return await self._call_tool("update_param_shape", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_dtype(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update only the dtype_desc field of parameters."""
        return await self._call_tool("update_param_dtype", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_dformat(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update only the dformat_desc field of parameters."""
        return await self._call_tool("update_param_dformat", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_optional(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update only the is_optional field of parameters."""
        return await self._call_tool("update_param_optional", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_src_content(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update only the src_content field of parameters."""
        return await self._call_tool("update_param_src_content", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_attrs(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update is_support_discontinuous and param_desc fields of parameters."""
        return await self._call_tool("update_param_attrs", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_allowed_range(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update only the allowed_range_value field of parameters."""
        return await self._call_tool("update_param_allowed_range", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_array_length(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update only the array_length field of parameters."""
        return await self._call_tool("update_param_array_length", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def update_param_constraint(self, doc_id: int, updates: list[dict]) -> dict:
        """Batch update only the param_constraint field of parameters."""
        return await self._call_tool("update_param_constraint", {
            "doc_id": doc_id,
            "updates": json.dumps(updates, ensure_ascii=False),
        })

    async def save_param_relations(self, doc_id: int, relations: list[dict]) -> dict:
        """Batch save parameter relations for a document version."""
        return await self._call_tool("save_relations", {
            "doc_id": doc_id,
            "relations": json.dumps(relations, ensure_ascii=False),
        })

    async def query_param_relations(self, doc_id: int) -> list[dict]:
        """Query parameter relations for a document version."""
        return await self._call_tool("query_relations", {"doc_id": doc_id})

    async def query_param_relations_by_operator(self, operator_name: str | None = None) -> list[dict]:
        """Query parameter relations, optionally filtered by operator name."""
        args: dict[str, Any] = {}
        if operator_name is not None:
            args["operator_name"] = operator_name
        return await self._call_tool("query_relations_by_operator", args)

    async def save_function_signatures(self, doc_id: int, signatures: list[dict]) -> dict:
        """Batch save function signatures for a document version."""
        return await self._call_tool("save_function_signatures", {
            "doc_id": doc_id,
            "signatures": json.dumps(signatures, ensure_ascii=False),
        })

    async def query_function_signatures_by_operator(self, operator_name: str | None = None) -> list[dict]:
        """Query function signatures, optionally filtered by operator name."""
        args: dict[str, Any] = {}
        if operator_name is not None:
            args["operator_name"] = operator_name
        return await self._call_tool("query_function_signatures_by_operator", args)

    async def save_platform_support(self, doc_id: int, platforms: list[dict]) -> dict:
        """Batch save platform support info for a document version."""
        return await self._call_tool("save_platform_support", {
            "doc_id": doc_id,
            "platforms": json.dumps(platforms, ensure_ascii=False),
        })

    async def save_function_explanation_summary(
        self, doc_id: int, summary: dict
    ) -> str:
        """Save function explanation summary for a document version."""
        return await self._call_tool(
            "save_function_explanation_summary", {
                "doc_id": doc_id,
                "summary": json.dumps(summary, ensure_ascii=False),
            }
        )

    async def get_function_explanation_summary(self, doc_id: int) -> dict:
        """Retrieve function explanation summary for a document version."""
        result = await self._call_tool("get_function_explanation_summary", {
            "doc_id": doc_id,
        })
        return result if isinstance(result, dict) else {}

    async def query_platform_support_by_operator(self, operator_name: str | None = None) -> list[dict]:
        """Query platform support info, optionally filtered by operator name."""
        args: dict[str, Any] = {}
        if operator_name is not None:
            args["operator_name"] = operator_name
        return await self._call_tool("query_platform_support_by_operator", args)

    async def save_return_codes(self, doc_id: int, return_codes: list[dict]) -> dict:
        """Batch save return codes for a document version."""
        return await self._call_tool("save_return_codes", {
            "doc_id": doc_id,
            "return_codes": json.dumps(return_codes, ensure_ascii=False),
        })

    async def query_return_codes_by_operator(self, operator_name: str | None = None) -> list[dict]:
        """Query return codes, optionally filtered by operator name."""
        args: dict[str, Any] = {}
        if operator_name is not None:
            args["operator_name"] = operator_name
        return await self._call_tool("query_return_codes_by_operator", args)

    async def save_determinism(self, doc_id: int, determinism_records: list[dict]) -> dict:
        """Batch save determinism records for a document version."""
        return await self._call_tool("save_determinism", {
            "doc_id": doc_id,
            "determinism_records": json.dumps(determinism_records, ensure_ascii=False),
        })

    async def query_determinism_by_operator(self, operator_name: str | None = None) -> list[dict]:
        """Query determinism records, optionally filtered by operator name."""
        args: dict[str, Any] = {}
        if operator_name is not None:
            args["operator_name"] = operator_name
        return await self._call_tool("query_determinism_by_operator", args)

    async def save_dtype_combinations(self, doc_id: int, combos: list[dict]) -> dict:
        """Batch save dtype combination records for a document version."""
        return await self._call_tool("save_dtype_combinations", {
            "doc_id": doc_id,
            "combos": json.dumps(combos, ensure_ascii=False),
        })

    async def query_dtype_combos_by_operator(self, operator_name: str | None = None) -> list[dict]:
        """Query dtype combination records, optionally filtered by operator name."""
        args: dict[str, Any] = {}
        if operator_name is not None:
            args["operator_name"] = operator_name
        return await self._call_tool("query_dtype_combos", args)

    async def query_function_signatures_by_doc_id(self, doc_id: int) -> list[dict]:
        """Query function signatures for a specific document version by doc_id."""
        return await self._call_tool("query_function_signatures_by_doc_id", {"doc_id": doc_id})

    async def query_platform_support_by_doc_id(self, doc_id: int) -> list[dict]:
        """Query platform support for a specific document version by doc_id."""
        return await self._call_tool("query_platform_support_by_doc_id", {"doc_id": doc_id})

    async def query_return_codes_by_doc_id(self, doc_id: int) -> list[dict]:
        """Query return codes for a specific document version by doc_id."""
        return await self._call_tool("query_return_codes_by_doc_id", {"doc_id": doc_id})

    async def query_dtype_combos_by_doc_id(self, doc_id: int) -> list[dict]:
        """Query dtype combinations for a specific document version by doc_id."""
        return await self._call_tool("query_dtype_combos_by_doc_id", {"doc_id": doc_id})

    async def save_constraints_result(
        self,
        doc_id: int,
        operator_name: str,
        product_support: str,
        platform_support: str,
        function_explanation: str,
        function_signature: str = "",
    ) -> dict:
        """Save assembled constraints result for a document version."""
        return await self._call_tool("save_constraints_result", {
            "doc_id": doc_id,
            "operator_name": operator_name,
            "product_support": product_support,
            "platform_support": platform_support,
            "function_explanation": function_explanation,
            "function_signature": function_signature,
        })

    async def query_constraints_result(self, operator_name: str | None = None) -> list[dict]:
        """Query constraints results, optionally filtered by operator name."""
        args: dict[str, Any] = {}
        if operator_name is not None:
            args["operator_name"] = operator_name
        return await self._call_tool("query_constraints_result", args)
