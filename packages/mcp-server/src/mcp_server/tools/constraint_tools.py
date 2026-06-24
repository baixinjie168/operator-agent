"""MCP Tools for constraint tools: domain-specific CRUD operations."""

from __future__ import annotations

import json

from mcp_server.db import get_db


def save_return_codes(doc_id: int, return_codes: list[dict]) -> dict:
    """Save return codes for a specific document version.

    Uses DELETE + INSERT for idempotency.

    Args:
        doc_id: Primary key of document_versions table.
        return_codes: List of dicts with keys: function_name, return_value,
                     error_code, descriptions (list of strings).

    Returns:
        dict with count of saved return codes.
    """
    db = get_db()
    conn = db.conn
    conn.execute("DELETE FROM return_codes WHERE doc_id = ?", (doc_id,))
    for rc in return_codes:
        conn.execute(
            "INSERT INTO return_codes "
            "(doc_id, function_name, return_value, error_code, descriptions, source_citation) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                rc.get("function_name", ""),
                rc.get("return_value", ""),
                rc.get("error_code", 0),
                json.dumps(rc.get("descriptions", []), ensure_ascii=False),
                rc.get("source_citation", ""),
            ),
        )
    conn.commit()
    return {"saved": len(return_codes)}


def query_return_codes_by_operator(operator_name: str | None = None) -> list[dict]:
    """Query return codes, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If None, returns all return codes.

    Returns:
        List of return code dicts with operator context.
    """
    db = get_db()
    conn = db.conn

    if operator_name:
        rows = conn.execute(
            "SELECT rc.id, o.name, dv.version, rc.function_name, rc.return_value, "
            "rc.error_code, rc.descriptions, rc.source_citation "
            "FROM return_codes rc "
            "JOIN document_versions dv ON rc.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "WHERE o.name = ? ORDER BY rc.function_name, rc.error_code",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT rc.id, o.name, dv.version, rc.function_name, rc.return_value, "
            "rc.error_code, rc.descriptions, rc.source_citation "
            "FROM return_codes rc "
            "JOIN document_versions dv ON rc.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "ORDER BY o.name, rc.function_name, rc.error_code",
        ).fetchall()

    return [
        {
            "id": r[0],
            "operator_name": r[1],
            "version": r[2],
            "function_name": r[3],
            "return_value": r[4],
            "error_code": r[5],
            "descriptions": json.loads(r[6]),
            "source_citation": r[7],
        }
        for r in rows
    ]


def query_return_codes_by_doc_id(doc_id: int) -> list[dict]:
    """Query return codes for a specific document version by doc_id."""
    db = get_db()
    rows = db.conn.execute(
        "SELECT id, function_name, return_value, error_code, descriptions, source_citation "
        "FROM return_codes WHERE doc_id = ? ORDER BY function_name, error_code",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "function_name": r[1],
            "return_value": r[2],
            "error_code": r[3],
            "descriptions": json.loads(r[4]) if r[4] else [],
            "source_citation": r[5],
        }
        for r in rows
    ]


def save_dtype_combinations(doc_id: int, combos: list[dict]) -> dict:
    """Save dtype combination records for a specific document version.

    Uses DELETE + INSERT for idempotency.

    Args:
        doc_id: Primary key of document_versions table.
        combos: List of dicts with keys: function_name, platform, combo.
                combo is a dict like {"x1": "FLOAT32", "x2": "FLOAT16"}.

    Returns:
        dict with count of saved records.
    """
    db = get_db()
    conn = db.conn
    conn.execute("DELETE FROM dtype_combinations WHERE doc_id = ?", (doc_id,))
    for record in combos:
        combo_data = record.get("combo", {})
        combo_json = (
            json.dumps(combo_data, ensure_ascii=False)
            if isinstance(combo_data, dict)
            else combo_data
        )
        conn.execute(
            "INSERT INTO dtype_combinations (doc_id, function_name, platform, combo) "
            "VALUES (?, ?, ?, ?)",
            (
                doc_id,
                record.get("function_name", ""),
                record.get("platform", "通用"),
                combo_json,
            ),
        )
    conn.commit()
    return {"saved": len(combos)}


def query_dtype_combos_by_operator(operator_name: str | None = None) -> list[dict]:
    """Query dtype combination records, optionally filtered by operator name.

    Args:
        operator_name: Optional operator name filter. If None, returns all records.

    Returns:
        List of dtype combination dicts with operator context.
    """
    db = get_db()
    conn = db.conn

    if operator_name:
        rows = conn.execute(
            "SELECT dc.id, o.name, dv.version, dc.function_name, dc.platform, dc.combo "
            "FROM dtype_combinations dc "
            "JOIN document_versions dv ON dc.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "WHERE o.name = ? ORDER BY dc.function_name, dc.platform, dc.id",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT dc.id, o.name, dv.version, dc.function_name, dc.platform, dc.combo "
            "FROM dtype_combinations dc "
            "JOIN document_versions dv ON dc.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "ORDER BY o.name, dc.function_name, dc.platform, dc.id",
        ).fetchall()

    return [
        {
            "id": r[0],
            "operator_name": r[1],
            "version": r[2],
            "function_name": r[3],
            "platform": r[4],
            "combo": json.loads(r[5]) if r[5] else {},
        }
        for r in rows
    ]


def query_dtype_combos_by_doc_id(doc_id: int) -> list[dict]:
    """Query dtype combinations for a specific document version by doc_id."""
    db = get_db()
    rows = db.conn.execute(
        "SELECT id, function_name, platform, combo "
        "FROM dtype_combinations WHERE doc_id = ? ORDER BY function_name, platform, id",
        (doc_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "function_name": r[1],
            "platform": r[2],
            "combo": json.loads(r[3]) if r[3] else {},
        }
        for r in rows
    ]


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
) -> dict:
    """Save assembled constraints result for a document version.

    Uses INSERT OR REPLACE for idempotency.

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
        dict with saved flag.
    """
    db = get_db()
    conn = db.conn
    conn.execute(
        "INSERT OR REPLACE INTO constraints_result "
        "(doc_id, operator_name, product_support, "
        "function_explanation, function_signature, return_codes, "
        "deterministic_computing, inputs, outputs, constraints_in_parameters, "
        "dtype_support_description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (doc_id, operator_name, product_support,
         function_explanation, function_signature, return_codes,
         deterministic_computing, inputs, outputs, constraints_in_parameters,
         dtype_support_description),
    )
    conn.commit()
    return {"saved": True}


def query_constraints_result(operator_name: str | None = None) -> list[dict]:
    """Query constraints results, optionally filtered by operator name."""
    db = get_db()
    conn = db.conn

    _cols = (
        "cr.id, cr.doc_id, cr.operator_name, dv.version, "
        "cr.product_support, cr.function_explanation, "
        "cr.function_signature, cr.return_codes, "
        "cr.deterministic_computing, cr.inputs, cr.outputs, "
        "cr.constraints_in_parameters, cr.dtype_support_description"
    )
    _base = "FROM constraints_result cr JOIN document_versions dv ON cr.doc_id = dv.id"

    if operator_name:
        rows = conn.execute(
            f"SELECT {_cols} {_base} WHERE cr.operator_name = ? ORDER BY dv.version DESC",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {_cols} {_base} ORDER BY cr.operator_name, dv.version DESC",
        ).fetchall()

    return [
        {
            "id": r[0],
            "doc_id": r[1],
            "operator_name": r[2],
            "version": r[3],
            "product_support": json.loads(r[4]) if r[4] else [],
            "function_explanation": (json.loads(r[5]) if r[5] else {}).get("description", ""),
            "function_detail": json.loads(r[5]) if r[5] else {},
            "function_signature": r[6] or "",
            "return_info": json.loads(r[7]) if r[7] else [],
            "deterministic_computing": json.loads(r[8]) if r[8] else {},
            "inputs": json.loads(r[9]) if r[9] else {},
            "outputs": json.loads(r[10]) if r[10] else {},
            "constraints_in_parameters": json.loads(r[11]) if r[11] else {},
            "dtype_support_description": json.loads(r[12]) if r[12] else {},
        }
        for r in rows
    ]


def save_json_constraints(doc_id: int, json_constraints: str) -> dict:
    """Save the final result.json structure to document_versions.json_constraints.

    Args:
        doc_id: Primary key of document_versions table.
        json_constraints: JSON string of the complete result.json structure.

    Returns:
        dict with saved flag.
    """
    db = get_db()
    conn = db.conn
    conn.execute(
        "UPDATE document_versions SET json_constraints = ? WHERE id = ?",
        (json_constraints, doc_id),
    )
    conn.commit()
    return {"saved": True}


def get_json_constraints(operator_name: str) -> dict | None:
    """Retrieve json_constraints from the latest document version for an operator.

    Args:
        operator_name: Operator name.

    Returns:
        Parsed JSON dict, or None if not found.
    """
    db = get_db()
    conn = db.conn
    row = conn.execute(
        "SELECT dv.json_constraints FROM document_versions dv "
        "JOIN operators o ON dv.operator_id = o.id "
        "WHERE o.name = ? ORDER BY dv.version DESC LIMIT 1",
        (operator_name,),
    ).fetchone()
    if not row or not row[0] or row[0] == "{}":
        return None
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None


def save_implicit_params(doc_id: int, mappings_json: str, rendered_text: str) -> dict:
    """Persist implicit (non-operator) parameters for traceability.

    Uses INSERT OR REPLACE for idempotency (one mapping set per doc_id).

    Args:
        doc_id: Primary key of document_versions table.
        mappings_json: JSON string of the implicit_params list.
        rendered_text: The rendered prompt context text that the LLM sees.

    Returns:
        dict with saved flag.
    """
    db = get_db()
    conn = db.conn
    conn.execute(
        "INSERT OR REPLACE INTO implicit_params "
        "(doc_id, mappings_json, rendered_text) "
        "VALUES (?, ?, ?)",
        (doc_id, mappings_json, rendered_text),
    )
    conn.commit()
    return {"saved": True}


def query_implicit_params_by_doc_id(doc_id: int) -> dict | None:
    """Query implicit parameters for a specific document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        Dict with mappings_json (parsed list) and rendered_text, or None if not found.
    """
    db = get_db()
    row = db.conn.execute(
        "SELECT mappings_json, rendered_text "
        "FROM implicit_params WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        return None
    try:
        mappings = json.loads(row[0]) if row[0] else []
    except (json.JSONDecodeError, TypeError):
        mappings = []
    return {
        "mappings": mappings,
        "rendered_text": row[1] or "",
    }


def save_parameter_representations(
    doc_id: int, representations_json: str,
) -> dict:
    """Persist parameter_representation records for a document version.

    Uses INSERT OR REPLACE for idempotency (one record per doc_id).

    Args:
        doc_id: Primary key of document_versions table.
        representations_json: JSON string with shape
            {"representations": [...], "platform_representations": {platform: [...]}}.

    Returns:
        dict with saved flag.
    """
    db = get_db()
    conn = db.conn
    conn.execute(
        "INSERT OR REPLACE INTO parameter_representations "
        "(doc_id, representations) VALUES (?, ?)",
        (doc_id, representations_json),
    )
    conn.commit()
    return {"saved": True}


def query_parameter_representations_by_doc_id(doc_id: int) -> dict | None:
    """Query parameter_representation records for a document version by doc_id.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        Dict with 'representations' (list) and 'platform_representations'
        (dict platform -> list), or None if not found.
    """
    db = get_db()
    row = db.conn.execute(
        "SELECT representations FROM parameter_representations WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        return None
    try:
        data = json.loads(row[0]) if row[0] else {}
    except (json.JSONDecodeError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "representations": data.get("representations", []) or [],
        "platform_representations": data.get("platform_representations", {}) or {},
    }
