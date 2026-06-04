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
    query_constraints_result as _query_constraints_result,
)
from mcp_server.tools.document_tools import (
    query_determinism_by_operator as _query_determinism_by_operator,
)
from mcp_server.tools.document_tools import (
    query_dtype_combos_by_doc_id as _query_dc_by_doc_id,
)
from mcp_server.tools.document_tools import (
    query_dtype_combos_by_operator as _query_dtype_combos_by_operator,
)
from mcp_server.tools.document_tools import (
    query_function_signatures as _query_function_signatures,
)
from mcp_server.tools.document_tools import (
    query_function_signatures_by_doc_id as _query_fn_sigs_by_doc_id,
)
from mcp_server.tools.document_tools import (
    query_param_relations as _query_param_relations,
)
from mcp_server.tools.document_tools import (
    query_param_relations_by_operator as _query_param_relations_by_operator,
)
from mcp_server.tools.document_tools import (
    query_platform_support as _query_platform_support,
)
from mcp_server.tools.document_tools import (
    query_platform_support_by_doc_id as _query_plat_by_doc_id,
)
from mcp_server.tools.document_tools import (
    query_return_codes_by_doc_id as _query_rc_by_doc_id,
)
from mcp_server.tools.document_tools import (
    query_return_codes_by_operator as _query_return_codes_by_operator,
)
from mcp_server.tools.document_tools import (
    save_constraints_result as _save_constraints_result,
)
from mcp_server.tools.document_tools import (
    get_function_explanation_summary as _get_fn_expl_summary,
)
from mcp_server.tools.document_tools import (
    save_function_explanation_summary as _save_fn_expl,
)
from mcp_server.tools.document_tools import (
    save_determinism as _save_determinism,
)
from mcp_server.tools.document_tools import (
    save_dtype_combinations as _save_dtype_combinations,
)
from mcp_server.tools.document_tools import (
    save_function_signatures as _save_function_signatures,
)
from mcp_server.tools.document_tools import (
    save_param_relations as _save_param_relations,
)
from mcp_server.tools.document_tools import (
    save_platform_support as _save_platform_support,
)
from mcp_server.tools.document_tools import (
    save_return_codes as _save_return_codes,
)
from mcp_server.tools.document_tools import (
    update_param_allowed_range as _update_param_allowed_range,
)
from mcp_server.tools.document_tools import (
    update_param_array_length as _update_param_array_length,
)
from mcp_server.tools.document_tools import (
    update_param_attrs as _update_param_attrs,
)
from mcp_server.tools.document_tools import (
    update_param_constraint as _update_param_constraint,
)
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
                 description, data_type, data_format, shape.

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
def update_param_attrs(doc_id: int, updates: str) -> str:
    """Batch update is_support_discontinuous and param_desc fields of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 is_support_discontinuous, param_desc.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_attrs(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_allowed_range(doc_id: int, updates: str) -> str:
    """Batch update only the allowed_range_value field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 allowed_range_value.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_allowed_range(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_array_length(doc_id: int, updates: str) -> str:
    """Batch update only the array_length field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name, array_length.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_array_length(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_constraint(doc_id: int, updates: str) -> str:
    """Batch update only the param_constraint field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 param_constraint.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_constraint(doc_id, data)
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


@mcp.tool()
def save_relations(doc_id: int, relations: str) -> str:
    """Batch save parameter relations for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        relations: JSON string — array of relation dicts.

    Returns:
        JSON string with count of saved relations.
    """
    data = json.loads(relations)
    result = _save_param_relations(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_relations(doc_id: int) -> str:
    """Query parameter relations for a document version.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON array of relation objects.
    """
    result = _query_param_relations(doc_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_relations_by_operator(operator_name: str | None = None) -> str:
    """Query parameter relations, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If null, returns all relations.

    Returns:
        JSON array of relation objects with operator context.
    """
    result = _query_param_relations_by_operator(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_function_signatures(doc_id: int, signatures: str) -> str:
    """Batch save function signatures for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        signatures: JSON string — array of signature dicts with keys:
                   function_name, return_type, parameters (list of {name, type}),
                   full_signature, raw_code.

    Returns:
        JSON string with count of saved signatures.
    """
    data = json.loads(signatures)
    result = _save_function_signatures(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_function_signatures_by_operator(operator_name: str | None = None) -> str:
    """Query function signatures, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If null, returns all signatures.

    Returns:
        JSON array of signature objects with operator context.
    """
    result = _query_function_signatures(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_platform_support(doc_id: int, platforms: str) -> str:
    """Batch save platform support info for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        platforms: JSON string — array of platform dicts with keys:
                  platform_name, is_supported.

    Returns:
        JSON string with count of saved platforms.
    """
    data = json.loads(platforms)
    result = _save_platform_support(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_platform_support_by_operator(operator_name: str | None = None) -> str:
    """Query platform support info, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If null, returns all platforms.

    Returns:
        JSON array of platform objects with operator context.
    """
    result = _query_platform_support(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_return_codes(doc_id: int, return_codes: str) -> str:
    """Batch save return codes for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        return_codes: JSON string — array of dicts with keys:
                     function_name, return_value, error_code, descriptions.

    Returns:
        JSON string with count of saved return codes.
    """
    data = json.loads(return_codes)
    result = _save_return_codes(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_return_codes_by_operator(operator_name: str | None = None) -> str:
    """Query return codes, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If null, returns all return codes.

    Returns:
        JSON array of return code objects with operator context.
    """
    result = _query_return_codes_by_operator(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_determinism(doc_id: int, determinism_records: str) -> str:
    """Batch save determinism records for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        determinism_records: JSON string — array of dicts with keys:
                            product, value, src_text.

    Returns:
        JSON string with count of saved records.
    """
    data = json.loads(determinism_records)
    result = _save_determinism(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_determinism_by_operator(operator_name: str | None = None) -> str:
    """Query determinism records, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If null, returns all records.

    Returns:
        JSON array of determinism objects with operator context.
    """
    result = _query_determinism_by_operator(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_dtype_combinations(doc_id: int, combos: str) -> str:
    """Batch save dtype combination records for a document version.

    Uses DELETE + INSERT for idempotency.

    Args:
        doc_id: Primary key of document_versions table.
        combos: JSON string — array of dicts with keys:
                function_name, platform, combo (a dict like {"x1": "FLOAT32", "x2": "FLOAT16"}).

    Returns:
        JSON string with count of saved records.
    """
    data = json.loads(combos)
    result = _save_dtype_combinations(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_dtype_combos(operator_name: str | None = None) -> str:
    """Query dtype combination records, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If null, returns all records.

    Returns:
        JSON array of dtype combination objects with operator context.
    """
    result = _query_dtype_combos_by_operator(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_function_signatures_by_doc_id(doc_id: int) -> str:
    """Query function signatures for a specific document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON array of function signature objects.
    """
    result = _query_fn_sigs_by_doc_id(doc_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_platform_support_by_doc_id(doc_id: int) -> str:
    """Query platform support for a specific document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON array of platform support objects with deterministic_computing.
    """
    result = _query_plat_by_doc_id(doc_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_return_codes_by_doc_id(doc_id: int) -> str:
    """Query return codes for a specific document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON array of return code objects.
    """
    result = _query_rc_by_doc_id(doc_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_dtype_combos_by_doc_id(doc_id: int) -> str:
    """Query dtype combinations for a specific document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON array of dtype combination objects.
    """
    result = _query_dc_by_doc_id(doc_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_constraints_result(
    doc_id: int,
    operator_name: str,
    product_support: str,
    platform_support: str,
    function_explanation: str,
    function_signature: str = "",
) -> str:
    """Save assembled constraints result for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        operator_name: Operator name.
        product_support: JSON string of product support list.
        platform_support: JSON string of supported platform name list.
        function_explanation: JSON string of function-grouped constraint data.
        function_signature: full_signature of the GetWorkspaceSize function.

    Returns:
        JSON string with saved flag.
    """
    result = _save_constraints_result(
        doc_id, operator_name, product_support, platform_support,
        function_explanation, function_signature,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_constraints_result(operator_name: str | None = None) -> str:
    """Query constraints results, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If null, returns all results.

    Returns:
        JSON array of constraints result objects.
    """
    result = _query_constraints_result(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_function_explanation_summary(doc_id: int, summary: str) -> str:
    """Save function explanation summary for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        summary: JSON string with description, formula, key_points, source_text.

    Returns:
        JSON string with saved flag.
    """
    data = json.loads(summary)
    result = _save_fn_expl(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_function_explanation_summary(doc_id: int) -> str:
    """Retrieve function explanation summary for a document version.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON string with keys: description, formula, key_points, source_text.
    """
    result = _get_fn_expl_summary(doc_id)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # Ensure DB is initialized at startup
    get_db()
    mcp.run()
