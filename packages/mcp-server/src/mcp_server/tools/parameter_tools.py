"""MCP Tools for parameter tools: domain-specific CRUD operations."""

from __future__ import annotations

import json

from mcp_server.db import get_db


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
            "direction, src_content, dtype_desc, dformat_desc, shape) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                param.get("function_name", ""),
                param.get("param_name", ""),
                param.get("param_type", ""),
                param.get("direction", ""),
                param.get("src_content", ""),
                param.get("data_type", ""),
                param.get("data_format", ""),
                param.get("shape", ""),
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
        "src_content, dtype_desc, dformat_desc, shape, is_optional, "
        "is_support_discontinuous, array_length, param_desc, allowed_range_value, "
        "param_constraint, llm_description "
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
            "data_type": r[6],
            "data_format": r[7],
            "shape": r[8],
            "is_optional": r[9],
            "is_support_discontinuous": r[10],
            "array_length": r[11],
            "param_desc": r[12],
            "allowed_range_value": r[13],
            "param_constraint": r[14],
            "llm_description": r[15],
        }
        for r in rows
    ]


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
            "p.param_type, p.direction, p.src_content, p.llm_description, "
            "p.dtype_desc, p.dformat_desc, p.shape, p.is_optional, "
            "p.is_support_discontinuous, p.array_length, p.param_desc, p.allowed_range_value "
            "FROM parameters p "
            "JOIN document_versions dv ON p.doc_id = dv.id "
            "JOIN operators o ON dv.operator_id = o.id "
            "WHERE o.name = ? ORDER BY p.function_name, p.direction, p.param_name",
            (operator_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT p.id, o.name, dv.version, p.function_name, p.param_name, "
            "p.param_type, p.direction, p.src_content, p.llm_description, "
            "p.dtype_desc, p.dformat_desc, p.shape, p.is_optional, "
            "p.is_support_discontinuous, p.array_length, p.param_desc, p.allowed_range_value "
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
            "llm_description": r[8],
            "data_type": r[9],
            "data_format": r[10],
            "shape": r[11],
            "is_optional": r[12],
            "is_support_discontinuous": r[13],
            "array_length": r[14],
            "param_desc": r[15],
            "allowed_range_value": r[16],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Generic batch update — replaces all single-field update_param_* functions
# ---------------------------------------------------------------------------

# Whitelist: field name → DB column name
_PARAM_FIELD_COLUMN_MAP = {
    "shape": "shape",
    "dtype": "dtype_desc",
    "dformat": "dformat_desc",
    "is_optional": "is_optional",
    "is_support_discontinuous": "is_support_discontinuous",
    "param_desc": "param_desc",
    "direction": "direction",
    "array_length": "array_length",
    "allowed_range_value": "allowed_range_value",
    "param_constraint": "param_constraint",
    "platform_attributes": "platform_attributes",
}

# Default values when the update dict doesn't contain the field key
_FIELD_DEFAULTS = {
    "is_support_discontinuous": '{"value":"N/A","src_text":""}',
    "array_length": "N/A",
    "allowed_range_value": "[]",
    "param_constraint": "{}",
    "platform_attributes": "{}",
    "is_optional": 0,
}

# Value key in the update dict (usually same as field name)
_FIELD_VALUE_KEY = {
    "shape": "shape",
    "dtype": "dtype",
    "dformat": "dformat",
    "is_optional": "is_optional",
    "is_support_discontinuous": "is_support_discontinuous",
    "param_desc": "param_desc",
    "direction": "direction",
    "array_length": "array_length",
    "allowed_range_value": "allowed_range_value",
    "param_constraint": "param_constraint",
    "platform_attributes": "platform_attributes",
}
def batch_update_param_field(doc_id: int, field: str, updates: list[dict]) -> dict:
    """Generic batch update for any parameter field.

    Args:
        doc_id: Primary key of document_versions table.
        field: Target field name (shape, dtype, dformat, is_optional, etc.).
        updates: List of dicts with function_name, param_name, and value.

    Returns:
        dict with count of updated parameters.

    Raises:
        ValueError: If field is not in the whitelist.
    """
    if field not in _PARAM_FIELD_COLUMN_MAP:
        raise ValueError(
            f"Unknown field '{field}'. "
            f"Allowed: {', '.join(sorted(_PARAM_FIELD_COLUMN_MAP))}"
        )

    column = _PARAM_FIELD_COLUMN_MAP[field]
    value_key = _FIELD_VALUE_KEY[field]
    default = _FIELD_DEFAULTS.get(field, "")

    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        value = u.get(value_key, default)
        cursor = conn.execute(
            f"UPDATE parameters SET {column} = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (value, doc_id, u.get("function_name", ""), u.get("param_name", "")),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


# ---------------------------------------------------------------------------
# Legacy single-field update wrappers (backward compat)
# ---------------------------------------------------------------------------

def update_param_shape(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the shape field of parameters."""
    return batch_update_param_field(doc_id, "shape", updates)


def update_param_dtype(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the dtype_desc field of parameters."""
    return batch_update_param_field(doc_id, "dtype", updates)


def update_param_dformat(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the dformat_desc field of parameters."""
    return batch_update_param_field(doc_id, "dformat", updates)


def update_param_optional(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the is_optional field of parameters."""
    return batch_update_param_field(doc_id, "is_optional", updates)


def update_param_attrs(doc_id: int, updates: list[dict]) -> dict:
    """Batch update is_support_discontinuous field of parameters."""
    return batch_update_param_field(doc_id, "is_support_discontinuous", updates)


def update_param_desc(doc_id: int, updates: list[dict]) -> dict:
    """Batch update param_desc field of parameters."""
    return batch_update_param_field(doc_id, "param_desc", updates)


def update_param_direction(doc_id: int, updates: list[dict]) -> dict:
    """Batch update direction field of parameters."""
    return batch_update_param_field(doc_id, "direction", updates)


def update_llm_descriptions(doc_id: int, updates: list[dict]) -> dict:
    """Batch update llm_description, src_content, direction, is_support_discontinuous, and description_audit.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name,
                 llm_description, src_content, direction, is_support_discontinuous,
                 description_audit.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        # Serialize description_audit to JSON string if it's a dict/list.
        # Without this, SQLite would receive a Python dict and raise
        # InterfaceError for unsupported types.
        audit = u.get("description_audit", "")
        if isinstance(audit, (dict, list)):
            audit = json.dumps(audit, ensure_ascii=False)

        cursor = conn.execute(
            "UPDATE parameters SET llm_description = ?, src_content = ?, "
            "direction = ?, is_support_discontinuous = ?, "
            "description_audit = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (
                u.get("llm_description", ""),
                u.get("src_content", ""),
                u.get("direction", ""),
                u.get("is_support_discontinuous", '{"value":"N/A","src_text":""}'),
                audit,
                doc_id,
                u.get("function_name", ""),
                u.get("param_name", ""),
            ),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_array_length(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the array_length field of parameters."""
    return batch_update_param_field(doc_id, "array_length", updates)


def update_param_allowed_range(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the allowed_range_value field of parameters."""
    return batch_update_param_field(doc_id, "allowed_range_value", updates)


def update_param_constraint(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the param_constraint field of parameters."""
    return batch_update_param_field(doc_id, "param_constraint", updates)


def update_param_platform_attributes(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the platform_attributes field of parameters."""
    return batch_update_param_field(doc_id, "platform_attributes", updates)
