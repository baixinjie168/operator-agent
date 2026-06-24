"""MCP Tools for platform tools: domain-specific CRUD operations."""

from __future__ import annotations

import json

from mcp_server.db import get_db


def save_platform_support(doc_id: int, platforms: list[dict]) -> dict:
    """Save platform support info for a specific document version.

    Uses UPSERT (INSERT ... ON CONFLICT DO UPDATE) for idempotency.
    Always updates is_supported; only updates deterministic_computing
    when the incoming value has a non-empty "value" field, to avoid
    overwriting determinism data already written by determinism_extract_node.

    Args:
        doc_id: Primary key of document_versions table.
        platforms: List of platform dicts with keys: platform_name, is_supported,
                   and optionally deterministic_computing.

    Returns:
        dict with count of saved platforms.
    """
    db = get_db()
    conn = db.conn
    default_det = {"value": "", "src_text": ""}
    for p in platforms:
        det = p.get("deterministic_computing", default_det)
        det_json = json.dumps(det, ensure_ascii=False)
        conn.execute(
            "INSERT INTO platform_support "
            "(doc_id, platform_name, is_supported, deterministic_computing) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(doc_id, platform_name) DO UPDATE SET "
            "is_supported = excluded.is_supported, "
            "deterministic_computing = CASE "
            "  WHEN json_extract(excluded.deterministic_computing, '$.value') != '' "
            "  THEN excluded.deterministic_computing "
            "  ELSE platform_support.deterministic_computing END",
            (
                doc_id,
                p.get("platform_name", ""),
                p.get("is_supported", 0),
                det_json,
            ),
        )
    conn.commit()
    return {"saved": len(platforms)}


def query_platform_support(operator_name: str | None = None) -> list[dict]:
    """Query platform support, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If None, returns all platforms.

    Returns:
        List of platform dicts with operator context, including deterministic_computing.
    """
    db = get_db()
    conn = db.conn
    _default_det = '{"value":"","src_text":""}'

    if operator_name:
        rows = conn.execute(
            "SELECT ps.id, o.name, dv.version, ps.platform_name, ps.is_supported, "
            "ps.deterministic_computing "
            "FROM platform_support ps "
            "JOIN document_versions dv ON ps.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "WHERE o.name = ? ORDER BY ps.platform_name",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ps.id, o.name, dv.version, ps.platform_name, ps.is_supported, "
            "ps.deterministic_computing "
            "FROM platform_support ps "
            "JOIN document_versions dv ON ps.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "ORDER BY o.name, ps.platform_name",
        ).fetchall()

    return [
        {
            "id": r[0],
            "operator_name": r[1],
            "version": r[2],
            "platform_name": r[3],
            "is_supported": r[4],
            "deterministic_computing": json.loads(r[5] or _default_det),
        }
        for r in rows
    ]


def query_platform_support_by_doc_id(doc_id: int) -> list[dict]:
    """Query platform support for a specific document version by doc_id."""
    db = get_db()
    _default_det = '{"value":"","src_text":""}'
    rows = db.conn.execute(
        "SELECT id, platform_name, is_supported, deterministic_computing "
        "FROM platform_support WHERE doc_id = ? ORDER BY platform_name",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "platform_name": r[1],
            "is_supported": r[2],
            "deterministic_computing": json.loads(r[3] or _default_det),
        }
        for r in rows
    ]


def save_determinism(doc_id: int, determinism_records: list[dict]) -> dict:
    """Save determinism records into platform_support.deterministic_computing.

    Uses UPSERT: if a platform_support row exists for (doc_id, product),
    update only its deterministic_computing column; otherwise insert a new
    row with is_supported defaulting to 0.

    Args:
        doc_id: Primary key of document_versions table.
        determinism_records: List of dicts with keys: product, value, src_text.

    Returns:
        dict with count of saved records.
    """
    db = get_db()
    conn = db.conn
    for record in determinism_records:
        product = record.get("product", "")
        value_str = "true" if record.get("value") else "false"
        src_text = record.get("src_text", "")
        det_json = json.dumps(
            {"value": value_str, "src_text": src_text},
            ensure_ascii=False,
        )
        conn.execute(
            "INSERT INTO platform_support "
            "(doc_id, platform_name, is_supported, deterministic_computing) "
            "VALUES (?, ?, 0, ?) "
            "ON CONFLICT(doc_id, platform_name) DO UPDATE SET "
            "deterministic_computing = excluded.deterministic_computing",
            (doc_id, product, det_json),
        )
    conn.commit()
    return {"saved": len(determinism_records)}


def query_determinism_by_operator(operator_name: str | None = None) -> list[dict]:
    """Query determinism records from platform_support.deterministic_computing,
    optionally filtered by operator name.

    Only returns rows where deterministic_computing.value is non-empty.
    Maps platform_name → product and value string → bool for backward compatibility.

    Args:
        operator_name: Optional operator name filter. If None, returns all records.

    Returns:
        List of determinism dicts with operator context.
    """
    db = get_db()
    conn = db.conn

    base_sql = (
        "SELECT ps.id, o.name, dv.version, ps.platform_name, "
        "ps.deterministic_computing "
        "FROM platform_support ps "
        "JOIN document_versions dv ON ps.doc_id = dv.id "
        "JOIN operators o ON dv.operator_id = o.id "
    )
    filter_clause = "json_extract(ps.deterministic_computing, '$.value') != ''"

    if operator_name:
        rows = conn.execute(
            base_sql + "WHERE o.name = ? AND " + filter_clause +
            " ORDER BY ps.platform_name",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            base_sql + "WHERE " + filter_clause +
            " ORDER BY o.name, ps.platform_name",
        ).fetchall()

    result = []
    for r in rows:
        det = json.loads(r[4])
        raw_val = det.get("value", "")
        # Support both new ("true"/"false") and legacy ("确定性"/"非确定性") formats
        is_det = raw_val == "true" or raw_val == "确定性"
        result.append({
            "id": r[0],
            "operator_name": r[1],
            "version": r[2],
            "product": r[3],
            "value": is_det,
            "src_text": det.get("src_text", ""),
        })
    return result


def save_platform_constants(doc_id: int, constants_json: str) -> dict:
    """Persist platform constants for traceability.

    Uses DELETE + INSERT for idempotency (one set per doc_id).

    Args:
        doc_id: Primary key of document_versions table.
        constants_json: JSON string of the platform_constants list.

    Returns:
        dict with saved flag and count.
    """
    db = get_db()
    conn = db.conn
    # Clear existing constants for this doc_id
    conn.execute(
        "DELETE FROM platform_constants WHERE doc_id = ?", (doc_id,),
    )
    try:
        constants = json.loads(constants_json) if constants_json else []
    except (json.JSONDecodeError, TypeError):
        constants = []
    for pc in constants:
        conn.execute(
            "INSERT INTO platform_constants "
            "(doc_id, const_name, description, platform_values, source_citation) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                doc_id,
                pc.get("const_name", ""),
                pc.get("description", ""),
                json.dumps(pc.get("platform_values", []), ensure_ascii=False),
                pc.get("source_citation", ""),
            ),
        )
    conn.commit()
    return {"saved": True, "count": len(constants)}


def query_platform_constants_by_doc_id(doc_id: int) -> dict | None:
    """Query platform constants for a specific document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        Dict with constants list, or None if not found.
    """
    db = get_db()
    rows = db.conn.execute(
        "SELECT const_name, description, platform_values, source_citation "
        "FROM platform_constants WHERE doc_id = ?",
        (doc_id,),
    ).fetchall()
    if not rows:
        return {"constants": []}
    constants = []
    for row in rows:
        try:
            pv = json.loads(row[2]) if row[2] else []
        except (json.JSONDecodeError, TypeError):
            pv = []
        constants.append({
            "const_name": row[0],
            "description": row[1] or "",
            "platform_values": pv,
            "source_citation": row[3] or "",
        })
    return {"constants": constants}
