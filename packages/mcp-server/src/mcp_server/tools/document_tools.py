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


def get_doc_for_constraint_check(doc_id: int) -> dict | None:
    """Retrieve raw content + json_constraints for constraint checking.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        dict with content, json_constraints, operator_name, doc_id, or None.
    """
    db = get_db()
    row = db.conn.execute(
        "SELECT dv.id, dv.content, dv.json_constraints, o.name "
        "FROM document_versions dv "
        "JOIN operators o ON dv.operator_id = o.id "
        "WHERE dv.id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "doc_id": row[0],
        "content": row[1] or "",
        "json_constraints": row[2] or "{}",
        "operator_name": row[3] or "",
    }


def get_doc_for_check_by_name(operator_name: str) -> dict | None:
    """Retrieve raw content + json_constraints by operator name (latest version).

    Args:
        operator_name: Operator name.

    Returns:
        dict with doc_id, content, json_constraints, operator_name, or None.
    """
    db = get_db()
    row = db.conn.execute(
        "SELECT dv.id, dv.content, dv.json_constraints, o.name "
        "FROM document_versions dv "
        "JOIN operators o ON dv.operator_id = o.id "
        "WHERE o.name = ? "
        "ORDER BY dv.version DESC LIMIT 1",
        (operator_name,),
    ).fetchone()
    if not row:
        return None
    return {
        "doc_id": row[0],
        "content": row[1] or "",
        "json_constraints": row[2] or "{}",
        "operator_name": row[3] or "",
    }


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


def get_sections_by_type(doc_id: int, section_type: str) -> list[dict]:
    """Retrieve ALL sections of a given section_type from parsed document.

    Unlike ``get_section_by_type`` (which returns only the first match), this
    returns every matching section — needed when an operator splits its
    parameter table across multiple same-type sections (e.g. H3-divided
    ``params_execute``). ``table_column_extract`` merges them so no row is
    lost to the 同型遮蔽 (first-match masking) bug.

    Args:
        doc_id: Primary key of document_versions table.
        section_type: The section_type to match (e.g. "params_execute").

    Returns:
        List of matching section dicts (empty if none / no parsed doc).
    """
    parsed = get_parsed_by_doc_id(doc_id)
    if not parsed:
        return []
    return [
        section for section in parsed.get("sections", [])
        if section.get("section_type") == section_type
    ]


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
        "SELECT o.id, o.name, o.source_url, o.created_at, "
        "  (SELECT MAX(dv.version) FROM document_versions dv WHERE dv.operator_id = o.id) AS latest_version "
        "FROM operators o ORDER BY o.name"
    ).fetchall()

    return [
        {
            "id": r[0],
            "name": r[1],
            "source_url": r[2],
            "created_at": r[3],
            "latest_version": r[4],
        }
        for r in rows
    ]


def delete_operator(operator_name: str) -> dict:
    """Delete an operator and ALL associated data (cascade delete).

    Deletion order:
    1. Find all doc_ids from document_versions for this operator
    2. Delete child table rows for each doc_id
    3. Delete document_versions rows
    4. Delete the operator record from operators table
    5. Clean up task_items references

    Args:
        operator_name: Name of the operator to delete.

    Returns:
        dict with deleted_operator, deleted_doc_versions count.
    """
    db = get_db()
    conn = db.conn

    # Find operator
    op = conn.execute(
        "SELECT id FROM operators WHERE name = ?", (operator_name,)
    ).fetchone()
    if op is None:
        raise ValueError(f"Operator '{operator_name}' not found")

    operator_id = op[0]

    # Collect all doc_ids for this operator
    doc_rows = conn.execute(
        "SELECT id FROM document_versions WHERE operator_id = ?", (operator_id,)
    ).fetchall()
    doc_ids = [r[0] for r in doc_rows]

    # Delete child table rows for each doc_id.
    # These tables all have a doc_id column referencing document_versions.id.
    child_tables = [
        "parameters",
        "param_relations",
        "function_signatures",
        "platform_support",
        "return_codes",
        "dtype_combinations",
        "constraints_result",
        "implicit_params",
        "platform_constants",
        "parameter_representations",
        "shape_dim_mappings",
        "pipeline_runs",
    ]
    for doc_id in doc_ids:
        # Collect run_ids from pipeline_runs BEFORE deleting them,
        # so we can clean up pipeline_events (which uses run_id, not doc_id).
        run_ids = [r[0] for r in conn.execute(
            "SELECT run_id FROM pipeline_runs WHERE doc_id = ?", (doc_id,)
        ).fetchall()]

        # Delete child table rows that reference doc_id
        for table in child_tables:
            conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))

        # Delete pipeline_events via run_id (no doc_id column in this table)
        for run_id in run_ids:
            conn.execute("DELETE FROM pipeline_events WHERE run_id = ?", (run_id,))

        # test_cases uses constraint_doc_id — clear references
        conn.execute(
            "UPDATE test_cases SET constraint_doc_id = NULL WHERE constraint_doc_id = ?",
            (doc_id,),
        )

    # Delete document_versions
    conn.execute("DELETE FROM document_versions WHERE operator_id = ?", (operator_id,))

    # Clean up task_items that reference these doc_ids
    for doc_id in doc_ids:
        conn.execute(
            "UPDATE task_items SET doc_id = NULL WHERE doc_id = ?", (doc_id,)
        )

    # Delete the operator record
    conn.execute("DELETE FROM operators WHERE id = ?", (operator_id,))

    conn.commit()

    return {
        "deleted_operator": operator_name,
        "deleted_doc_versions": len(doc_ids),
    }


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
        "param_constraint, llm_description, usage_notes "
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
            "usage_notes": r[16],
        }
        for r in rows
    ]


def save_constraint_check_report(doc_id: int, report_html: str) -> dict:
    """Save constraint check HTML report to document_versions.

    Args:
        doc_id: Primary key of document_versions table.
        report_html: Full HTML report string.

    Returns:
        dict with saved status and doc_id.
    """
    db = get_db()
    conn = db.conn
    conn.execute(
        "UPDATE document_versions SET constraint_check_report = ? WHERE id = ?",
        (report_html, doc_id),
    )
    conn.commit()
    return {"saved": True, "doc_id": doc_id}


def get_constraint_check_report(doc_id: int) -> dict:
    """Retrieve constraint check HTML report from document_versions.

    Args:
        doc_id: Primary key of document_versions table.

    Returns:
        dict with report key (HTML string or None).
    """
    db = get_db()
    row = db.conn.execute(
        "SELECT constraint_check_report FROM document_versions WHERE id = ?",
        (doc_id,),
    ).fetchone()
    return {"report": row[0] if row else None}


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


def update_param_attrs(doc_id: int, updates: list[dict]) -> dict:
    """Batch update is_support_discontinuous field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name,
                 is_support_discontinuous.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET is_support_discontinuous = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (
                u.get("is_support_discontinuous", '{"value":"N/A","src_text":""}'),
                doc_id,
                u.get("function_name", ""),
                u.get("param_name", ""),
            ),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_desc(doc_id: int, updates: list[dict]) -> dict:
    """Batch update param_desc field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, param_desc.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET param_desc = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (
                u.get("param_desc", ""),
                doc_id,
                u.get("function_name", ""),
                u.get("param_name", ""),
            ),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_direction(doc_id: int, updates: list[dict]) -> dict:
    """Batch update direction field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, direction.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET direction = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (
                u.get("direction", ""),
                doc_id,
                u.get("function_name", ""),
                u.get("param_name", ""),
            ),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


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
    """Batch update only the array_length field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, array_length.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET array_length = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (u.get("array_length", "N/A"), doc_id, u.get("function_name", ""), u.get("param_name", "")),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_allowed_range(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the allowed_range_value field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, allowed_range_value.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET allowed_range_value = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (u.get("allowed_range_value", "[]"), doc_id, u.get("function_name", ""), u.get("param_name", "")),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


def update_param_constraint(doc_id: int, updates: list[dict]) -> dict:
    """Batch update only the param_constraint field of parameters.

    Args:
        doc_id: Primary key of document_versions table.
        updates: List of dicts with keys: function_name, param_name, param_constraint.

    Returns:
        dict with count of updated parameters.
    """
    db = get_db()
    conn = db.conn
    count = 0
    for u in updates:
        cursor = conn.execute(
            "UPDATE parameters SET param_constraint = ? "
            "WHERE doc_id = ? AND function_name = ? AND param_name = ?",
            (
                u.get("param_constraint", "{}"),
                doc_id,
                u.get("function_name", ""),
                u.get("param_name", ""),
            ),
        )
        count += cursor.rowcount
    conn.commit()
    return {"updated": count}


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


def update_json_constraints_by_name(operator_name: str, json_constraints: str) -> dict:
    """Update json_constraints for the latest document version of an operator.

    Args:
        operator_name: Operator name.
        json_constraints: JSON string of the updated constraints.

    Returns:
        dict with saved flag and doc_id.
    """
    db = get_db()
    conn = db.conn
    row = conn.execute(
        "SELECT dv.id FROM document_versions dv "
        "JOIN operators o ON dv.operator_id = o.id "
        "WHERE o.name = ? ORDER BY dv.version DESC LIMIT 1",
        (operator_name,),
    ).fetchone()
    if not row:
        return {"saved": False, "error": f"Operator '{operator_name}' not found"}
    doc_id = row[0]
    conn.execute(
        "UPDATE document_versions SET json_constraints = ? WHERE id = ?",
        (json_constraints, doc_id),
    )
    conn.commit()
    return {"saved": True, "doc_id": doc_id}


def get_document_content(operator_name: str, version: int | None = None) -> dict | None:
    """Retrieve raw Markdown content from the latest document version for an operator.

    Args:
        operator_name: Operator name.
        version: Version number (defaults to latest).

    Returns:
        dict with content and version, or None if not found.
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
        row = conn.execute(
            "SELECT content, version FROM document_versions "
            "WHERE operator_id = ? ORDER BY version DESC LIMIT 1",
            (operator_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT content, version FROM document_versions "
            "WHERE operator_id = ? AND version = ?",
            (operator_id, version),
        ).fetchone()

    if not row or not row[0]:
        return None

    return {"content": row[0], "version": row[1], "operator_name": operator_name}


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
