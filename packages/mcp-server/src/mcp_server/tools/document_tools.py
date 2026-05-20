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
        dict with keys: status ("new"/"unchanged"/"updated"), version (int or None)
    """
    db = get_db()
    conn = db.conn

    row = conn.execute(
        "SELECT id FROM operators WHERE name = ?", (operator_name,)
    ).fetchone()

    if not row:
        return {"status": "new", "version": None}

    operator_id = row[0]

    latest = conn.execute(
        "SELECT version, content_hash FROM document_versions "
        "WHERE operator_id = ? ORDER BY version DESC LIMIT 1",
        (operator_id,),
    ).fetchone()

    if not latest:
        return {"status": "new", "version": None}

    if latest[1] == content_hash:
        return {"status": "unchanged", "version": latest[0]}

    return {"status": "updated", "version": latest[0]}


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

    conn.execute(
        "INSERT INTO document_versions (operator_id, version, content, content_hash) "
        "VALUES (?, ?, ?, ?)",
        (operator_id, next_version, content, content_hash),
    )
    conn.commit()

    return {"operator_id": operator_id, "version": next_version}


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


def save_parameters(operator_name: str, version: int, parameters: list[dict]) -> dict:
    """Save parsed parameters for a specific operator version.

    Uses INSERT OR REPLACE for idempotency (upsert on unique constraint).

    Args:
        operator_name: Operator name.
        version: Document version number.
        parameters: List of parameter dicts matching ParsedParameter fields.

    Returns:
        dict with count of saved parameters.
    """
    db = get_db()
    conn = db.conn

    operator_row = conn.execute(
        "SELECT id FROM operators WHERE name = ?", (operator_name,)
    ).fetchone()
    if not operator_row:
        return {"saved": 0, "error": f"Operator '{operator_name}' not found"}

    operator_id = operator_row[0]

    for param in parameters:
        conn.execute(
            "INSERT OR REPLACE INTO parameters "
            "(operator_id, version, function_name, param_name, param_type, "
            "direction, description, usage_notes, data_type, data_format, shape, attributes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                operator_id,
                version,
                param.get("function_name", ""),
                param.get("param_name", ""),
                param.get("param_type", ""),
                param.get("direction", "input"),
                param.get("description", ""),
                param.get("usage_notes", ""),
                param.get("data_type", ""),
                param.get("data_format", ""),
                param.get("shape", ""),
                json.dumps(param.get("attributes", {}), ensure_ascii=False),
            ),
        )

    conn.commit()
    return {"saved": len(parameters)}
