"""MCP Server for operator-agent subsystem.

Provides tools for document version checking, parsing, and persistence.
Run via: python -m mcp_server.server
"""

import json

from mcp.server.fastmcp import FastMCP

from mcp_server.db import get_db
from mcp_server.tools.document_tools import (
    check_document_version,
    do_save_product_support,
    get_parsed_document,
    get_section_by_type,
    list_all_operators,
    parse_document,
    query_parameters,
    query_params_by_doc_id,
    save_document,
    save_parameters,
    save_parsed_document,
    update_param_descriptions,
)
from mcp_server.tools.document_tools import get_parsed_by_doc_id as _get_parsed_by_doc_id
from mcp_server.tools.document_tools import (
    update_param_dformat as _update_param_dformat,
)
from mcp_server.tools.document_tools import (
    update_param_dtype as _update_param_dtype,
)
from mcp_server.tools.document_tools import (
    update_param_optional as _update_param_optional,
)
from mcp_server.tools.document_tools import (
    update_param_shape as _update_param_shape,
)
from mcp_server.tools.document_tools import (
    update_param_src_content as _update_param_src_content,
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
def get_parsed_by_doc_id(doc_id: int) -> str:
    """Retrieve parsed document data by document_versions primary key.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON string of ParsedOperatorDocument, or "null" if not found.
    """
    result = _get_parsed_by_doc_id(doc_id)
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
def save_params(doc_id: int, parameters: str) -> str:
    """Save parsed parameters for a specific document version.

    Args:
        doc_id: Primary key of document_versions table.
        parameters: JSON string — array of parameter dicts.

    Returns:
        JSON string with count of saved parameters.
    """
    params = json.loads(parameters)
    result = save_parameters(doc_id, params)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_product_support(doc_id: int, product_support_data: str) -> str:
    """Save product support data for a specific document version.

    Args:
        doc_id: Primary key of document_versions table.
        product_support_data: JSON string — array of {product, support} dicts.

    Returns:
        "ok" on success.
    """
    data = json.loads(product_support_data)
    do_save_product_support(doc_id, data)
    return "ok"


@mcp.resource("operator://{name}")
def get_operator(name: str) -> str:
    """Get operator info and latest parsed data."""
    result = get_parsed_document(name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_section(doc_id: int, section_type: str) -> str:
    """Retrieve a specific section from a parsed document by section_type.

    Args:
        doc_id: Primary key of document_versions table.
        section_type: Section type to match (e.g. "params_get_workspace").

    Returns:
        JSON string of the section dict, or "null" if not found.
    """
    result = get_section_by_type(doc_id, section_type)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_params_by_doc(doc_id: int) -> str:
    """Query parameters for a specific document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON array of parameter objects.
    """
    result = query_params_by_doc_id(doc_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_descs(doc_id: int, updates: str) -> str:
    """Batch update parameter description fields.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 description, usage_notes, data_type, data_format, shape, memory_desc.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = update_param_descriptions(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_shape(doc_id: int, updates: str) -> str:
    """Batch update only the shape field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name, shape.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_shape(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_dtype(doc_id: int, updates: str) -> str:
    """Batch update only the dtype_desc field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name, dtype.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_dtype(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_dformat(doc_id: int, updates: str) -> str:
    """Batch update only the dformat_desc field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name, dformat.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_dformat(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_optional(doc_id: int, updates: str) -> str:
    """Batch update only the is_optional field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name, is_optional.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_optional(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_src_content(doc_id: int, updates: str) -> str:
    """Batch update only the src_content field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name, src_content.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_src_content(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_params(operator_name: str | None = None) -> str:
    """Query parameters from the database, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If null, returns all parameters.

    Returns:
        JSON array of parameter objects.
    """
    result = query_parameters(operator_name)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # Ensure DB is initialized at startup
    get_db()
    mcp.run()
