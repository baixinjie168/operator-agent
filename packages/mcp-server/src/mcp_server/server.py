"""MCP Server for operator-agent subsystem.

Provides tools for document version checking, parsing, and persistence.
Run via: python -m mcp_server.server
"""

import json

from mcp.server.fastmcp import FastMCP

from mcp_server.db import get_db
from mcp_server.tools.document_tools import (
    check_document_version,
    get_parsed_document,
    list_all_operators,
    parse_document,
    save_document,
    save_parameters,
    save_parsed_document,
)

mcp = FastMCP("operator-agent-mcp-server")


@mcp.tool()
def check_version(operator_name: str, content_hash: str) -> str:
    """Check if a document has already been parsed or is a new/updated version.

    Args:
        operator_name: The operator name (e.g. 'aclnnBatchNormElemt').
        content_hash: SHA256 hash of the document content.

    Returns:
        JSON string with: status ("new"/"unchanged"/"updated"), version (int or null)
    """
    result = check_document_version(operator_name, content_hash)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_doc(operator_name: str, content: str, source_url: str | None = None) -> str:
    """Save a document to the database and return its operator_id and version.

    Args:
        operator_name: The operator name.
        content: Full Markdown content of the document.
        source_url: Optional source URL.

    Returns:
        JSON string with: operator_id (int), version (int)
    """
    result = save_document(operator_name, content, source_url)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def parse_doc(content: str) -> str:
    """Parse a CANN operator Markdown document into structured sections.

    Returns classified sections, product support, and function signatures.

    Args:
        content: Full Markdown content of the operator document.

    Returns:
        JSON string of ParsedOperatorDocument.
    """
    result = parse_document(content)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_parsed(operator_name: str, version: int | None = None) -> str:
    """Retrieve a previously parsed document from the database.

    Args:
        operator_name: Operator name.
        version: Version number (null for latest).

    Returns:
        JSON string of ParsedOperatorDocument, or "null" if not found.
    """
    result = get_parsed_document(operator_name, version)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_parsed(operator_name: str, version: int, parsed_data: str) -> str:
    """Save parsed document data to the database.

    Args:
        operator_name: Operator name.
        version: Version number.
        parsed_data: JSON string of ParsedOperatorDocument.

    Returns:
        "ok" on success.
    """
    data = json.loads(parsed_data)
    save_parsed_document(operator_name, version, data)
    return "ok"


@mcp.resource("operator://list")
def list_operators_resource() -> str:
    """List all registered operators."""
    db = get_db()
    rows = db.conn.execute("SELECT name, source_url FROM operators").fetchall()
    return json.dumps([{"name": r[0], "source_url": r[1]} for r in rows], ensure_ascii=False)


@mcp.tool()
def query_operators() -> str:
    """List all registered operators with their latest version info.

    Returns:
        JSON array of operators with name, source_url, latest_version, created_at.
    """
    result = list_all_operators()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_params(operator_name: str, version: int, parameters: str) -> str:
    """Save parsed parameters for a specific operator version.

    Args:
        operator_name: Operator name.
        version: Document version number.
        parameters: JSON string — array of parameter dicts.

    Returns:
        JSON string with count of saved parameters.
    """
    params = json.loads(parameters)
    result = save_parameters(operator_name, version, params)
    return json.dumps(result, ensure_ascii=False)


@mcp.resource("operator://{name}")
def get_operator(name: str) -> str:
    """Get operator info and latest parsed data."""
    result = get_parsed_document(name)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # Ensure DB is initialized at startup
    get_db()
    mcp.run()
