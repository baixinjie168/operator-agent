"""MCP Tools for signature tools: domain-specific CRUD operations."""

from __future__ import annotations

import json

from mcp_server.db import get_db


def save_function_signatures(doc_id: int, signatures: list[dict]) -> dict:
    """Save function signatures for a specific document version.

    Uses DELETE + INSERT for idempotency.

    Args:
        doc_id: Primary key of document_versions table.
        signatures: List of signature dicts with keys: function_name, return_type,
                   parameters (list of {name, type}), full_signature, raw_code.

    Returns:
        dict with count of saved signatures.
    """
    db = get_db()
    conn = db.conn
    conn.execute("DELETE FROM function_signatures WHERE doc_id = ?", (doc_id,))
    for sig in signatures:
        conn.execute(
            "INSERT INTO function_signatures "
            "(doc_id, function_name, return_type, parameters, full_signature, raw_code) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                sig.get("function_name", ""),
                sig.get("return_type", ""),
                json.dumps(sig.get("parameters", []), ensure_ascii=False),
                sig.get("full_signature", ""),
                sig.get("raw_code", ""),
            ),
        )
    conn.commit()
    return {"saved": len(signatures)}


def query_function_signatures(operator_name: str | None = None) -> list[dict]:
    """Query function signatures, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If None, returns all signatures.

    Returns:
        List of signature dicts with operator context.
    """
    db = get_db()
    conn = db.conn

    if operator_name:
        rows = conn.execute(
            "SELECT fs.id, o.name, dv.version, fs.function_name, fs.return_type, "
            "fs.parameters, fs.full_signature, fs.raw_code "
            "FROM function_signatures fs "
            "JOIN document_versions dv ON fs.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "WHERE o.name = ? ORDER BY fs.function_name",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT fs.id, o.name, dv.version, fs.function_name, fs.return_type, "
            "fs.parameters, fs.full_signature, fs.raw_code "
            "FROM function_signatures fs "
            "JOIN document_versions dv ON fs.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "ORDER BY o.name, fs.function_name",
        ).fetchall()

    return [
        {
            "id": r[0],
            "operator_name": r[1],
            "version": r[2],
            "function_name": r[3],
            "return_type": r[4],
            "parameters": json.loads(r[5]),
            "full_signature": r[6],
            "raw_code": r[7],
        }
        for r in rows
    ]


def query_function_signatures_by_doc_id(doc_id: int) -> list[dict]:
    """Query function signatures for a specific document version by doc_id."""
    db = get_db()
    rows = db.conn.execute(
        "SELECT id, function_name, return_type, parameters, full_signature, raw_code "
        "FROM function_signatures WHERE doc_id = ? ORDER BY function_name",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "function_name": r[1],
            "return_type": r[2],
            "parameters": json.loads(r[3]) if r[3] else [],
            "full_signature": r[4],
            "raw_code": r[5],
        }
        for r in rows
    ]
