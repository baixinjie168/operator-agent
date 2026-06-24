"""MCP Tools for relation tools: domain-specific CRUD operations."""

from __future__ import annotations

import json

from mcp_server.db import get_db


def update_param_relation_objects(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the relation_object field of param_relations.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: id, relation_object.

    Returns:
        dict with count of updated rows.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE param_relations SET relation_object = ? "
            "WHERE id = ? AND doc_id = ?",
            (
                u.get("relation_object", "{}"),
                u["id"],
                doc_id,
            ),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def save_param_relations(doc_id: int, relations: list[dict]) -> dict:
    db = get_db()
    conn = db.conn
    conn.execute("DELETE FROM param_relations WHERE doc_id = ?", (doc_id,))
    for r in relations:
        conn.execute(
            "INSERT INTO param_relations "
            "(doc_id, function_name, relation_type, platform, "
            "description, params, param_optional, source_citation, relation_object) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                r.get("function_name", ""),
                r.get("relation_type", ""),
                r.get("platform", ""),
                r.get("description", ""),
                json.dumps(r.get("params", []), ensure_ascii=False),
                json.dumps(r.get("param_optional", {}), ensure_ascii=False),
                r.get("source_citation", ""),
                json.dumps(r.get("relation_object", {}), ensure_ascii=False),
            ),
        )
    conn.commit()
    return {"saved": len(relations)}


def query_param_relations(doc_id: int) -> list[dict]:
    db = get_db()
    rows = db.conn.execute(
        "SELECT id, function_name, relation_type, platform, "
        "description, params, param_optional, source_citation, relation_object "
        "FROM param_relations WHERE doc_id = ? ORDER BY id",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "function_name": r[1],
            "relation_type": r[2],
            "platform": r[3],
            "description": r[4],
            "params": json.loads(r[5]),
            "param_optional": json.loads(r[6]),
            "source_citation": r[7],
            "relation_object": json.loads(r[8] or "{}"),
        }
        for r in rows
    ]


def query_param_relations_by_operator(operator_name: str | None = None) -> list[dict]:
    """Query parameter relations, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If None, returns all relations.

    Returns:
        List of relation dicts with operator context.
    """
    db = get_db()
    conn = db.conn

    if operator_name:
        rows = conn.execute(
            "SELECT pr.id, o.name, dv.version, pr.function_name, pr.relation_type, "
            "pr.platform, pr.description, pr.params, pr.param_optional, pr.source_citation, "
            "pr.relation_object "
            "FROM param_relations pr "
            "JOIN document_versions dv ON pr.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "WHERE o.name = ? ORDER BY pr.function_name, pr.id",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT pr.id, o.name, dv.version, pr.function_name, pr.relation_type, "
            "pr.platform, pr.description, pr.params, pr.param_optional, pr.source_citation, "
            "pr.relation_object "
            "FROM param_relations pr "
            "JOIN document_versions dv ON pr.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "ORDER BY o.name, pr.function_name, pr.id",
        ).fetchall()

    return [
        {
            "id": r[0],
            "operator_name": r[1],
            "version": r[2],
            "function_name": r[3],
            "relation_type": r[4],
            "platform": r[5],
            "description": r[6],
            "params": json.loads(r[7]),
            "param_optional": json.loads(r[8]),
            "source_citation": r[9],
            "relation_object": json.loads(r[10] or "{}"),
        }
        for r in rows
    ]
