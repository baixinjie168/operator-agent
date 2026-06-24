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
)
from mcp_server.tools.document_tools import (
    get_document_content as _get_document_content,
)
from mcp_server.tools.document_tools import (
    get_function_explanation_summary as _get_fn_expl_summary,
)
from mcp_server.tools.document_tools import (
    get_json_constraints as _get_json_constraints,
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
    save_json_constraints as _save_json_constraints,
)
from mcp_server.tools.document_tools import (
    get_json_constraints as _get_json_constraints,
)
from mcp_server.tools.constraint_tools import (
    save_implicit_params as _save_implicit_params,
)
from mcp_server.tools.constraint_tools import (
    query_implicit_params_by_doc_id as _query_ip_by_doc_id,
)
from mcp_server.tools.constraint_tools import (
    save_parameter_representations as _save_parameter_representations,
)
from mcp_server.tools.constraint_tools import (
    query_parameter_representations_by_doc_id as _query_pr_by_doc_id,
)
from mcp_server.tools.platform_tools import (
    save_platform_constants as _save_platform_constants,
)
from mcp_server.tools.platform_tools import (
    query_platform_constants_by_doc_id as _query_pc_by_doc_id,
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
    save_function_explanation_summary as _save_fn_expl,
)
from mcp_server.tools.document_tools import (
    save_function_signatures as _save_function_signatures,
)
from mcp_server.tools.document_tools import (
    save_json_constraints as _save_json_constraints,
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
    update_json_constraints_by_name as _update_json_constraints_by_name,
)
from mcp_server.tools.parameter_tools import (
    batch_update_param_field as _batch_update_param_field,
)
from mcp_server.tools.parameter_tools import (
    update_param_allowed_range as _update_param_allowed_range,
)
from mcp_server.tools.document_tools import (
    update_param_array_length as _update_param_array_length,
)
from mcp_server.tools.document_tools import (
    update_param_attrs as _update_param_attrs,
)
from mcp_server.tools.document_tools import (
    update_param_desc as _update_param_desc,
)
from mcp_server.tools.document_tools import (
    update_param_direction as _update_param_direction,
)
from mcp_server.tools.document_tools import (
    update_llm_descriptions as _update_llm_descriptions,
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
    update_param_relation_objects as _update_param_relation_objects,
)
from mcp_server.tools.document_tools import (
    update_param_shape as _update_param_shape,
)
from mcp_server.tools.parameter_tools import (
    update_param_platform_attributes as _update_param_plat_attrs,
)
from mcp_server.tools.parameter_tools import (
    update_param_usage_notes as _update_param_usage_notes,
)
from mcp_server.tools.task_tools import (
    create_task as _create_task,
)
from mcp_server.tools.task_tools import (
    create_task_items as _create_task_items,
)
from mcp_server.tools.task_tools import (
    get_pending_task_items as _get_pending_task_items,
)
from mcp_server.tools.task_tools import (
    get_task as _get_task,
)
from mcp_server.tools.task_tools import (
    get_task_with_items as _get_task_with_items,
)
from mcp_server.tools.task_tools import (
    list_tasks as _list_tasks,
)
from mcp_server.tools.task_tools import (
    refresh_task_progress as _refresh_task_progress,
)
from mcp_server.tools.task_tools import (
    reset_stuck_task_items as _reset_stuck_task_items,
)
from mcp_server.tools.task_tools import (
    update_task_item_status as _update_task_item_status,
)
from mcp_server.tools.task_tools import (
    update_task_status as _update_task_status,
)
from mcp_server.tools.test_case_tools import (
    do_get_test_cases as _do_get_test_cases,
)
from mcp_server.tools.test_case_tools import (
    do_list_test_case_operators as _do_list_test_case_operators,
)
from mcp_server.tools.test_case_tools import (
    do_save_test_cases as _do_save_test_cases,
)
from mcp_server.tools.task_tools import (
    delete_task as _delete_task,
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
        JSON string {"status": "ok"} on success.
    """
    data = json.loads(parsed_data)
    save_parsed_document(operator_name, version, data)
    return json.dumps({"status": "ok"})


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
        JSON string {"status": "ok"} on success.
    """
    data = json.loads(product_support_data)
    do_save_product_support(doc_id, data)
    return json.dumps({"status": "ok"})


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
def batch_update_params(doc_id: int, field: str, updates: str) -> str:
    """Generic batch update for any parameter field.

    Replaces the individual update_param_shape, update_param_dtype, etc. tools.

    Args:
        doc_id: Primary key of document_versions table.
        field: Target field name. One of: shape, dtype, dformat, is_optional,
               is_support_discontinuous, param_desc, direction, array_length,
               allowed_range_value, param_constraint.
        updates: JSON string — array of dicts with function_name, param_name,
                 and the field-specific value key.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _batch_update_param_field(doc_id, field, data)
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
def update_param_platform_attributes(doc_id: int, updates: str) -> str:
    """Batch update the platform_attributes field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 platform_attributes (JSON string of {field: {platform: value}}).

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_plat_attrs(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_usage_notes(doc_id: int, updates: str) -> str:
    """Batch update the usage_notes field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 usage_notes (JSON string of {platform: value}).

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_usage_notes(doc_id, data)
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
def update_param_attrs(doc_id: int, updates: str) -> str:
    """Batch update is_support_discontinuous field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 is_support_discontinuous.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_attrs(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_desc(doc_id: int, updates: str) -> str:
    """Batch update param_desc field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 param_desc.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_desc(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_param_direction(doc_id: int, updates: str) -> str:
    """Batch update direction field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 direction ('input' or 'output').

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_param_direction(doc_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_llm_descriptions(doc_id: int, updates: str) -> str:
    """Batch update llm_description, src_content, direction, and is_support_discontinuous.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with function_name, param_name,
                 llm_description, src_content, direction, is_support_discontinuous.

    Returns:
        JSON string with count of updated parameters.
    """
    data = json.loads(updates)
    result = _update_llm_descriptions(doc_id, data)
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
def update_param_relation_objects(doc_id: int, updates: str) -> str:
    """Batch update the relation_object field of param_relations.

    Args:
        doc_id: Primary key of document_versions table.
        updates: JSON string — array of dicts with id, relation_object.

    Returns:
        JSON string with count of updated rows.
    """
    data = json.loads(updates)
    result = _update_param_relation_objects(doc_id, data)
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
    function_explanation: str,
    function_signature: str = "",
    return_codes: str = "[]",
    deterministic_computing: str = "{}",
    inputs: str = "{}",
    outputs: str = "{}",
    constraints_in_parameters: str = "{}",
    dtype_support_description: str = "{}",
) -> str:
    """Save assembled constraints result for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        operator_name: Operator name.
        product_support: JSON string of supported platform name list.
        function_explanation: JSON string of function-grouped constraint data.
        function_signature: full_signature of the GetWorkspaceSize function.
        return_codes: JSON string of transformed return codes array.
        deterministic_computing: JSON string of {platform: {value, src_text}}.
        inputs: JSON string of {param_name: {platform: constraint}}.
        outputs: JSON string of {param_name: {platform: constraint}}.
        constraints_in_parameters: JSON string of {platform: [relation_object]}.
        dtype_support_description: JSON string of {platform: [combo]}.

    Returns:
        JSON string with saved flag.
    """
    result = _save_constraints_result(
        doc_id, operator_name, product_support,
        function_explanation, function_signature, return_codes,
        deterministic_computing, inputs, outputs, constraints_in_parameters,
        dtype_support_description,
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
def save_json_constraints(doc_id: int, json_constraints: str) -> str:
    """Save the final result.json structure to document_versions.json_constraints.

    Args:
        doc_id: Primary key of document_versions table.
        json_constraints: JSON string of the complete result.json structure.

    Returns:
        JSON string with saved flag.
    """
    result = _save_json_constraints(doc_id, json_constraints)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_document_content(operator_name: str, version: int | None = None) -> str:
    """Retrieve raw Markdown content from the latest document version for an operator.

    Args:
        operator_name: Operator name.
        version: Version number (null for latest).

    Returns:
        JSON string with content, version, operator_name, or null if not found.
    """
    result = _get_document_content(operator_name, version)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_json_constraints(operator_name: str) -> str:
    """Retrieve json_constraints from the latest document version for an operator.

    Args:
        operator_name: Operator name.

    Returns:
        JSON string of the json_constraints field, or null if not found.
    """
    result = _get_json_constraints(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_json_constraints_by_name(operator_name: str, json_constraints: str) -> str:
    """Update json_constraints for the latest document version of an operator.

    Args:
        operator_name: Operator name.
        json_constraints: JSON string of the updated constraints.

    Returns:
        JSON string with saved flag and doc_id.
    """
    result = _update_json_constraints_by_name(operator_name, json_constraints)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_implicit_params(doc_id: int, mappings_json: str, rendered_text: str) -> str:
    """Persist implicit (non-operator) parameters for a document version (traceability).

    Args:
        doc_id: Primary key of document_versions table.
        mappings_json: JSON string — array of implicit parameter mapping dicts.
        rendered_text: The rendered prompt context text that the LLM sees.

    Returns:
        JSON string with saved flag.
    """
    result = _save_implicit_params(doc_id, mappings_json, rendered_text)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_implicit_params_by_doc_id(doc_id: int) -> str:
    """Query implicit parameters for a document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON string with mappings (array) and rendered_text, or null if not found.
    """
    result = _query_ip_by_doc_id(doc_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_platform_constants(doc_id: int, constants_json: str) -> str:
    """Persist platform constants (external constants like rankSize) for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        constants_json: JSON string — array of platform constant dicts.

    Returns:
        JSON string with saved flag and count.
    """
    result = _save_platform_constants(doc_id, constants_json)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_platform_constants_by_doc_id(doc_id: int) -> str:
    """Query platform constants for a document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON string with constants array, or empty if not found.
    """
    result = _query_pc_by_doc_id(doc_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def save_parameter_representations(
    doc_id: int, representations_json: str,
) -> str:
    """Persist parameter_representation records for a document version.

    Args:
        doc_id: Primary key of document_versions table.
        representations_json: JSON string with shape
            {"representations": [...tensor-dim reps...],
             "platform_representations": {platform: [...constant reps...]}}.

    Returns:
        JSON string with saved flag.
    """
    result = _save_parameter_representations(doc_id, representations_json)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_parameter_representations_by_doc_id(doc_id: int) -> str:
    """Query parameter_representation records for a document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        JSON string with representations list and platform_representations
        dict, or empty if not found.
    """
    result = _query_pr_by_doc_id(doc_id)
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


# ── GeneratorAgent MCP tools ─────────────────────────────────────────────────


@mcp.tool()
def save_test_cases(
    operator_name: str,
    cases_json: str,
    source: str = "generated",
    output_dir: str | None = None,
) -> str:
    """Persist generated test cases to DB and ``cases/{operator_name}_cases.json``.

    Args:
        operator_name: Operator name (e.g. ``aclnnAdaLayerNorm``).
        cases_json: JSON-serialized list of test case records.
        source: Provenance label (default ``"generated"``).
        output_dir: Override for the cases directory.  ``None`` → ``cases/``
            under the project root.

    Returns:
        JSON with ``operator_name``, ``saved_count``, and absolute ``output_path``.
    """
    result = _do_save_test_cases(operator_name, cases_json, source=source, output_dir=output_dir)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def create_task(name: str, total_count: int, upload_dir: str) -> str:
    """Create a new batch processing task.

    Args:
        name: Task name (e.g. 'batch-20260604').
        total_count: Total number of documents in the task.
        upload_dir: Directory path where uploaded files are stored.

    Returns:
        JSON string with task_id and status.
    """
    result = _create_task(name, total_count, upload_dir)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def create_task_items(task_id: int, items: str) -> str:
    """Batch create task items for a task.

    Args:
        task_id: Parent task ID.
        items: JSON string — array of dicts with seq, operator_name, file_path.

    Returns:
        JSON string with count of inserted items.
    """
    data = json.loads(items)
    result = _create_task_items(task_id, data)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_task_status(task_id: int, status: str) -> str:
    """Update task status.

    Args:
        task_id: Task ID.
        status: New status (pending/running/completed/failed).

    Returns:
        JSON string with updated flag.
    """
    result = _update_task_status(task_id, status)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def update_task_item_status(
    item_id: int,
    status: str,
    error: str | None = None,
    doc_id: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> str:
    """Update task item status and optional fields.

    Args:
        item_id: Task item ID.
        status: New status (pending/running/completed/failed).
        error: Error message (if failed).
        doc_id: Document version ID (if completed).
        started_at: ISO timestamp when processing started.
        finished_at: ISO timestamp when processing finished.

    Returns:
        JSON string with updated flag.
    """
    result = _update_task_item_status(
        item_id, status, error=error, doc_id=doc_id,
        started_at=started_at, finished_at=finished_at,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_test_cases(operator_name: str) -> str:
    """Return the most recent saved test cases for ``operator_name``.

    Args:
        operator_name: Operator name.

    Returns:
        JSON with ``operator_name`` and ``cases`` list, or ``"null"`` if none.
    """
    result = _do_get_test_cases(operator_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_pending_task_items(task_id: int) -> str:
    """Get all pending task items for a task, ordered by seq.

    Args:
        task_id: Task ID.

    Returns:
        JSON array of task item dicts with status='pending'.
    """
    result = _get_pending_task_items(task_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def list_test_case_operators() -> str:
    """List operator names that have saved test cases, with counts.

    Returns:
        JSON array of ``{operator_name, count, last_created_at}`` objects.
    """
    result = _do_list_test_case_operators()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_task(task_id: int) -> str:
    """Get a single task by ID.

    Args:
        task_id: Task ID.

    Returns:
        JSON string of task dict, or "null" if not found.
    """
    result = _get_task(task_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def list_tasks() -> str:
    """List all tasks ordered by created_at DESC.

    Returns:
        JSON array of task dicts.
    """
    result = _list_tasks()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_task_with_items(task_id: int) -> str:
    """Get a task with all its items.

    Args:
        task_id: Task ID.

    Returns:
        JSON string of task dict with items array, or "null" if not found.
    """
    result = _get_task_with_items(task_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def refresh_task_progress(task_id: int) -> str:
    """Recount completed/failed items and update task progress.

    Args:
        task_id: Task ID.

    Returns:
        JSON string with updated completed_count and failed_count.
    """
    result = _refresh_task_progress(task_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def reset_stuck_task_items(task_id: int) -> str:
    """Reset task items stuck in 'running' back to 'pending'.

    Handles the case where the server crashed while items were being
    processed, leaving them in 'running' status indefinitely.
    Also resets the parent task status from 'running' to 'pending'.

    Args:
        task_id: Task ID.

    Returns:
        JSON string with count of reset items.
    """
    result = _reset_stuck_task_items(task_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def delete_task(task_id: int) -> str:
    """Delete a task and all associated operator data (cascade).

    Deletes: task items, document_versions, parameters, param_relations,
    function_signatures, platform_support, return_codes, dtype_combinations,
    constraints_result, implicit_params, platform_constants.

    Only allows deletion of finished tasks (not running).

    Args:
        task_id: Task ID to delete.

    Returns:
        JSON string with deleted_task_id, deleted_docs, deleted_items.
    """
    result = _delete_task(task_id)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # Ensure DB is initialized at startup
    get_db()
    mcp.run()
