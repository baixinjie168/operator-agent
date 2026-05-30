"""MCP Tools for document management: version checking, saving, and parsing."""

from __future__ import annotations

import hashlib
import json

from mcp_server.db import get_db
from mcp_server.parsers.document_parser import parse_operator_document


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


def save_parameters(doc_id: int, parameters: list[dict]) -> dict:
    """Save parsed parameters for a specific document version.

    Uses INSERT OR REPLACE for idempotency (upsert on unique constraint).

    Args:
        doc_id: Primary key of document_versions table.
        parameters: List of parameter dicts.

    Returns:
        dict with count of saved parameters.
    """
    db = get_db()
    conn = db.conn

    for param in parameters:
        conn.execute(
            "INSERT OR REPLACE INTO parameters "
            "(doc_id, function_name, param_name, param_type, "
            "direction, src_content, description, usage_notes, dtype_desc, dformat_desc, shape, memory_desc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                param.get("function_name", ""),
                param.get("param_name", ""),
                param.get("param_type", ""),
                param.get("direction", "input"),
                param.get("src_content", ""),
                param.get("description", ""),
                param.get("usage_notes", ""),
                param.get("data_type", ""),
                param.get("data_format", ""),
                param.get("shape", ""),
                param.get("memory_desc", ""),
            ),
        )

    conn.commit()
    return {"saved": len(parameters)}


def query_params_by_doc_id(doc_id: int) -> list[dict]:
    """Query parameters for a specific document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        List of parameter dicts for the given doc_id.
    """
    db = get_db()
    conn = db.conn
    rows = conn.execute(
        "SELECT id, function_name, param_name, param_type, direction, "
        "src_content, description, usage_notes, dtype_desc, dformat_desc, shape, memory_desc, is_optional "
        "FROM parameters WHERE doc_id = ? ORDER BY function_name, direction, param_name",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "function_name": r[1],
            "param_name": r[2],
            "param_type": r[3],
            "direction": r[4],
            "src_content": r[5],
            "description": r[6],
            "usage_notes": r[7],
            "data_type": r[8],
            "data_format": r[9],
            "shape": r[10],
            "memory_desc": r[11],
            "is_optional": r[12],
        }
        for r in rows
    ]


def update_param_descriptions(doc_id: int, updates: list[dict]) -> dict:
    """Batch update parameter description fields.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name,
                 direction, usage_notes, data_type, data_format, shape, memory_desc, description.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET direction = ?, description = ?, usage_notes = ?, "
            "dtype_desc = ?, dformat_desc = ?, shape = ?, memory_desc = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (
                u.get("direction", ""),
                u.get("description", ""),
                u.get("usage_notes", ""),
                u.get("data_type", ""),
                u.get("data_format", ""),
                u.get("shape", ""),
                u.get("memory_desc", ""),
                doc_id,
                u.get("function_name", ""),
                u.get("param_name", ""),
            ),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_shape(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the shape field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, shape.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET shape = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (u.get("shape", ""), doc_id, u.get("function_name", ""), u.get("param_name", "")),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_dtype(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the dtype_desc field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, dtype.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET dtype_desc = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (u.get("dtype", ""), doc_id, u.get("function_name", ""), u.get("param_name", "")),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_dformat(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the dformat_desc field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, dformat.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET dformat_desc = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (u.get("dformat", ""), doc_id, u.get("function_name", ""), u.get("param_name", "")),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_optional(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the is_optional field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, is_optional.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET is_optional = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (u.get("is_optional", 0), doc_id, u.get("function_name", ""), u.get("param_name", "")),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_src_content(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the src_content field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, src_content.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET src_content = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (u.get("src_content", ""), doc_id, u.get("function_name", ""), u.get("param_name", "")),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def query_parameters(operator_name: str | None = None) -> list[dict]:
    """Query parameters from the database, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If None, returns all parameters.

    Returns:
        List of parameter dicts with operator context.
    """
    db = get_db()
    conn = db.conn

    if operator_name:
        rows = conn.execute(
            "SELECT p.id, o.name, dv.version, p.function_name, p.param_name, "
            "p.param_type, p.direction, p.src_content, p.description, p.usage_notes, "
            "p.dtype_desc, p.dformat_desc, p.shape, p.memory_desc, p.is_optional "
            "FROM parameters p "
            "JOIN document_versions dv ON p.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "WHERE o.name = ? ORDER BY p.function_name, p.direction, p.param_name",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT p.id, o.name, dv.version, p.function_name, p.param_name, "
            "p.param_type, p.direction, p.src_content, p.description, p.usage_notes, "
            "p.dtype_desc, p.dformat_desc, p.shape, p.memory_desc, p.is_optional "
            "FROM parameters p "
            "JOIN document_versions dv ON p.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "ORDER BY o.name, p.function_name, p.direction, p.param_name",
        ).fetchall()

    return [
        {
            "id": r[0],
            "operator_name": r[1],
            "version": r[2],
            "function_name": r[3],
            "param_name": r[4],
            "param_type": r[5],
            "direction": r[6],
            "src_content": r[7],
            "description": r[8],
            "usage_notes": r[9],
            "data_type": r[10],
            "data_format": r[11],
            "shape": r[12],
            "memory_desc": r[13],
            "is_optional": r[14],
        }
        for r in rows
    ]
