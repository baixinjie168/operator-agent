"""Upload route: accepts operator documents and triggers agent-based processing."""

from __future__ import annotations

import hashlib
import json
import logging
import re

from langchain_core.messages import HumanMessage, ToolMessage
from starlette.requests import Request

from fastapi import APIRouter, UploadFile

from agent.graph import create_operator_agent
from agent.mcp_client import MCPClient
from agent.schemas.upload import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])

_mcp_client = MCPClient()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    """Upload a CANN operator Markdown document for processing.

    Flow:
    1. Read file content, compute hash, extract operator name
    2. Check version via MCP — return existing if unchanged
    3. Save new/updated document via MCP
    4. Invoke DeepAgent to parse document and sections
    5. Return results
    """
    content = (await file.read()).decode("utf-8")
    filename = file.filename or "unknown"

    operator_name = _extract_operator_name(content)
    if not operator_name:
        return UploadResponse(success=False, error=f"Cannot parse operator name from {filename}")

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    client = _mcp_client

    try:
        # Deterministic pre-processing
        version_info = await client.check_version(operator_name, content_hash)
        status = version_info.get("status", "new")
        existing_version = version_info.get("version")

        if status == "unchanged":
            existing = await client.get_parsed(operator_name, existing_version)
            if existing:
                return UploadResponse(
                    success=True,
                    operator_name=operator_name,
                    cann_version=existing.get("cann_version"),
                    status="unchanged",
                    version=existing_version,
                    sections_count=len(existing.get("sections", [])),
                )

        save_result = await client.save_doc(operator_name, content)
        new_version = save_result["version"]

        # Agent-based parsing
        parsed = await _invoke_parse_agent(operator_name, content, new_version)

        return UploadResponse(
            success=True,
            operator_name=parsed.get("operator_name", operator_name),
            cann_version=parsed.get("cann_version"),
            status=status,
            version=new_version,
            sections_count=len(parsed.get("sections", [])),
        )

    except Exception as e:
        logger.exception("Upload processing failed for %s", filename)
        return UploadResponse(success=False, error=str(e))


async def _invoke_parse_agent(
    operator_name: str,
    content: str,
    version: int,
) -> dict:
    """Invoke the DeepAgent to parse the document and save results.

    Falls back to direct MCP parsing on agent failure.
    """
    graph = create_operator_agent()

    user_message = (
        f"Process the operator document '{operator_name}' (version {version}).\n\n"
        f"Document content:\n{content}\n\n"
        f"After parsing, save the results with operator_name='{operator_name}' "
        f"and version={version}."
    )

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": f"upload_{operator_name}_{version}"}},
        )

        # Try to read back saved results from MCP
        saved = await _mcp_client.get_parsed(operator_name, version)
        if saved:
            return saved

        # If agent didn't save, extract parse_document result from tool messages
        parsed = _extract_tool_result(result, "parse_document")
        if parsed:
            await _mcp_client.save_parsed(operator_name, version, parsed)
            return parsed

        logger.warning("Agent completed but produced no parse results, falling back")
    except Exception:
        logger.exception("Agent invocation failed, falling back to direct MCP parse")

    # Fallback: direct MCP parse (the original behavior)
    parsed = await _mcp_client.parse_doc(content)
    await _mcp_client.save_parsed(operator_name, version, parsed)
    return parsed


def _extract_tool_result(result: dict, tool_name: str) -> dict | None:
    """Extract a tool call result from the agent's message history."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            content = msg.content
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    return None
            if isinstance(content, dict):
                return content
    return None


def _extract_operator_name(content: str) -> str | None:
    """Extract operator name from the H1 title line.

    Format: # {name}-CANN社区版{version}-昇腾社区
    """
    for line in content.split("\n"):
        m = re.match(r"^#\s+(.+?)-CANN社区版", line)
        if m:
            return m.group(1).strip()
    return None
