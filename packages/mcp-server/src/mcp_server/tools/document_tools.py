"""MCP Tools for document management: version checking, saving, and parsing.

This module retains core document operations and re-exports all domain-specific
functions from their dedicated modules for backward compatibility.
"""

from __future__ import annotations

import hashlib
import json

from mcp_server.db import get_db
from mcp_server.parsers.document_parser import parse_operator_document

# Re-export from domain-specific modules for backward compatibility
from mcp_server.tools.parameter_tools import (  # noqa: F401
    batch_update_param_field,
    query_params_by_doc_id,
    query_parameters,
    save_parameters,
    update_llm_descriptions,
    update_param_allowed_range,
    update_param_array_length,
    update_param_attrs,
    update_param_constraint,
    update_param_desc,
    update_param_direction,
    update_param_dformat,
    update_param_dtype,
    update_param_optional,
    update_param_platform_attributes,
    update_param_shape,
    update_param_usage_notes,
)
from mcp_server.tools.relation_tools import (  # noqa: F401
    query_param_relations,
    query_param_relations_by_operator,
    save_param_relations,
    update_param_relation_objects,
)
from mcp_server.tools.signature_tools import (  # noqa: F401
    query_function_signatures,
    query_function_signatures_by_doc_id,
    save_function_signatures,
)
from mcp_server.tools.platform_tools import (  # noqa: F401
    query_determinism_by_operator,
    query_platform_constants_by_doc_id,
    query_platform_support,
    query_platform_support_by_doc_id,
    save_determinism,
    save_platform_constants,
    save_platform_support,
)
from mcp_server.tools.constraint_tools import (  # noqa: F401
    get_json_constraints,
    query_constraints_result,
    query_dtype_combos_by_doc_id,
    query_dtype_combos_by_operator,
    query_return_codes_by_doc_id,
    query_return_codes_by_operator,
    query_implicit_params_by_doc_id,
    query_parameter_representations_by_doc_id,
    save_constraints_result,
    save_dtype_combinations,
    save_json_constraints,
    save_return_codes,
    save_implicit_params,
    save_parameter_representations,
)


def _compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def check_document_version(operator_name: str, content_hash: str) -> dict:
    """Check if a document has already been parsed or is a new/updated version.

    Args:
        operator_name: The operator name extracted from the document.
        content_hash: SHA256 hash of the document content.

    Returns:
        dict with keys: status, version, doc_id (int or None).
    """
    db = get_db()
    conn = db.conn

    row = conn.execute(
        "SELECT id FROM operators WHERE name = ?", (operator_name,)
    ).fetchone()

    if not row:
        return {"status": "new", "version": None, "doc_id": None}

    operator_id = row[0]

    latest = conn.execute(
        "SELECT id, version, content_hash FROM document_versions "
        "WHERE operator_id = ? ORDER BY version DESC LIMIT 1",
        (operator_id,),
    ).fetchone()

    if not latest:
        return {"status": "new", "version": None, "doc_id": None}

    if latest[2] == content_hash:
        return {"status": "unchanged", "version": latest[1], "doc_id": latest[0]}

    return {"status": "updated", "version": latest[1], "doc_id": latest[0]}


def save_document(operator_name: str, content: str, source_url: str | None = None) -> dict:
    """Save a document and return its operator_id and version number.

    Creates the operator record if it doesn't exist, then saves a new version.
    """
    db = get_db()
    conn = db.conn
    content_hash = _compute_hash(content)

    # Upsert operator
    conn.execute(
        "INSERT INTO operators (name, source_url) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET source_url = excluded.source_url",
        (operator_name, source_url),
    )

    operator_row = conn.execute(
        "SELECT id FROM operators WHERE name = ?", (operator_name,)
    ).fetchone()
    operator_id = operator_row[0]

    # Determine next version
    max_ver = conn.execute(
        "SELECT MAX(version) FROM document_versions WHERE operator_id = ?",
        (operator_id,),
    ).fetchone()[0]
    next_version = (max_ver or 0) + 1

    cursor = conn.execute(
        "INSERT INTO document_versions (operator_id, version, content, content_hash) "
        "VALUES (?, ?, ?, ?)",
        (operator_id, next_version, content, content_hash),
    )
    doc_id = cursor.lastrowid
    conn.commit()

    return {"operator_id": operator_id, "version": next_version, "doc_id": doc_id}


def parse_document(content: str) -> dict:
    """Parse a CANN operator Markdown document into structured data.

    Returns the ParsedOperatorDocument as a JSON-serializable dict.
    """
    parsed = parse_operator_document(content)
    return json.loads(parsed.model_dump_json())


def get_parsed_document(operator_name: str, version: int | None = None) -> dict | None:
    """Retrieve a previously parsed document from the database.

    Args:
        operator_name: Operator name.
        version: Version number (defaults to latest).

    Returns:
        ParsedOperatorDocument as dict, or None if not found.
    """
    db = get_db()
    conn = db.conn

    operator_row = conn.execute(
        "SELECT id FROM operators WHERE name = ?", (operator_name,)
    ).fetchone()
    if not operator_row:
        return None

    operator_id = operator_row[0]

    if version is None:
        ver_row = conn.execute(
            "SELECT parsed_data FROM document_versions "
            "WHERE operator_id = ? ORDER BY version DESC LIMIT 1",
            (operator_id,),
        ).fetchone()
    else:
        ver_row = conn.execute(
            "SELECT parsed_data FROM document_versions "
            "WHERE operator_id = ? AND version = ?",
            (operator_id, version),
        ).fetchone()

    if not ver_row or not ver_row[0]:
        return None

    return json.loads(ver_row[0])


def get_parsed_by_doc_id(doc_id: int) -> dict | None:
    """Retrieve parsed document data by document_versions.id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        ParsedOperatorDocument as dict, or None if not found.
    """
    db = get_db()
    row = db.conn.execute(
        "SELECT parsed_data FROM document_versions WHERE id = ?", (doc_id,)
    ).fetchone()
    if not row or not row[0]:
        return None
    return json.loads(row[0])


def get_section_by_type(doc_id: int, section_type: str) -> dict | None:
    """Retrieve a specific section from parsed document data by section_type.

    Args:
        doc_id: Primary key of document_versions table.
        section_type: The section_type to match (e.g. "params_get_workspace").

    Returns:
        The matching section dict, or None if not found.
    """
    parsed = get_parsed_by_doc_id(doc_id)
    if not parsed:
        return None
    for section in parsed.get("sections", []):
        if section.get("section_type") == section_type:
            return section
    return None


def save_parsed_document(operator_name: str, version: int, parsed_data: dict) -> None:
    """Save parsed document data to the database."""
    db = get_db()
    conn = db.conn

    operator_row = conn.execute(
        "SELECT id FROM operators WHERE name = ?", (operator_name,)
    ).fetchone()
    if not operator_row:
        return

    conn.execute(
        "UPDATE document_versions SET parsed_data = ? "
        "WHERE operator_id = ? AND version = ?",
        (json.dumps(parsed_data, ensure_ascii=False), operator_row[0], version),
    )
    conn.commit()


def do_save_product_support(doc_id: int, product_support_data: list[dict]) -> None:
    """Save product support data to the document_versions.product_support column."""
    db = get_db()
    conn = db.conn
    conn.execute(
        "UPDATE document_versions SET product_support = ? WHERE id = ?",
        (json.dumps(product_support_data, ensure_ascii=False), doc_id),
    )
    conn.commit()


def save_function_explanation_summary(doc_id: int, summary: dict) -> dict:
    """Save function explanation summary to document_versions."""
    db = get_db()
    conn = db.conn
    conn.execute(
        "UPDATE document_versions "
        "SET function_explanation_summary = ? WHERE id = ?",
        (json.dumps(summary, ensure_ascii=False), doc_id),
    )
    conn.commit()
    return {"saved": True}


def get_function_explanation_summary(doc_id: int) -> dict:
    """Retrieve function_explanation_summary from document_versions by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        Parsed JSON dict (keys: description, formula, key_points, source_text),
        or empty dict if not found.
    """
    db = get_db()
    row = db.conn.execute(
        "SELECT function_explanation_summary FROM document_versions WHERE id = ?",
        (doc_id,),
    ).fetchone()
    if not row or not row[0] or row[0] == "{}":
        return {}
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return {}


def list_all_operators() -> list[dict]:
    """List all registered operators with their latest version."""
    db = get_db()
    conn = db.conn

    rows = conn.execute(
        "SELECT o.name, o.source_url, o.created_at, "
        "  (SELECT MAX(dv.version) FROM document_versions dv WHERE dv.operator_id = o.id) AS latest_version "
        "FROM operators o ORDER BY o.name"
    ).fetchall()

    return [
        {
            "name": r[0],
            "source_url": r[1],
            "created_at": r[2],
            "latest_version": r[3],
        }
        for r in rows
    ]

