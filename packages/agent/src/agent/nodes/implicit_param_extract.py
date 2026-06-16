"""ImplicitParamExtract node: identify implicit dimension variables from document text.

Implicit dimension variables (e.g. hidden_size, input_size) appear as
named dimensions in shape descriptions (e.g. (num_layers, batch_size,
hidden_size)) but are NOT parameters in any function signature.

This node:
1. Scans raw section HTML text for shape tuples containing named variables
2. Filters out variables that already exist in function signatures
3. Injects the remaining (implicit) variables into the parameters table
4. Returns them via state["implicit_params"] for downstream consumption

Zero LLM calls - purely deterministic regex-based extraction.

Position in pipeline:
    table_column_extract -> **this node** -> llm_description_extract
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.state import PipelineState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Match shape tuples: Chinese or ASCII parentheses
_SHAPE_TUPLE_RE = re.compile(r"[（(]\s*([^)）]+)\s*[）)]")

# Named dimension variable: lowercase snake_case identifier
_DIM_VAR_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*)\b")

# Words that look like dimension variables but are not.
#
# Organised by category.  When adding new entries, keep them sorted
# within each group so duplicates are easy to spot.
_EXCLUDE_WORDS = frozenset({
    # -- Python / JSON literals --
    "true", "false", "none", "null",

    # -- Common non-dimension terms --
    "shape", "dtype", "format", "type",
    "input", "output", "tensor", "optional",
    "nd", "acl",

    # -- C base type keywords (no digit suffix) --
    "float", "double", "void", "char", "int", "long", "short",
    "signed", "unsigned", "struct", "union", "enum",

    # -- C fixed-width type keywords (with _t suffix) --
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "ptrdiff_t",

    # -- Bare data-type names (with digit suffix, already existed) --
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool", "string",

    # -- Common non-variable identifiers from Markdown / URLs --
    "common", "md", "html", "http", "https", "www",
    "aclnn", "aclrt", "device", "host",

    # -- English stop-words / logical operators --
    "or", "and", "if", "else", "when",
    "the", "for", "not", "with", "from",
})


def _is_markdown_url(content: str) -> bool:
    """Return True if *content* looks like a Markdown hyperlink URL.

    ``_SHAPE_TUPLE_RE`` happily matches the ``(url)`` part of
    ``[text](url)`` links.  Such matches are never shape tuples and
    should be skipped to avoid extracting spurious identifiers like
    ``common`` or ``md`` from paths such as ``../common/xxx.md``.
    """
    stripped = content.strip()
    if "/" in stripped:
        return True
    if stripped.endswith(".md") or stripped.endswith(".html"):
        return True
    if stripped.startswith("..") or stripped.startswith("./"):
        return True
    return False


_SECTION_TYPES = ("params_get_workspace", "params_execute")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case.

    Examples:
        numLayers -> num_layers
        batchSizeOptional -> batch_size_optional
    """
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\g<1>_\g<2>", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\g<1>_\g<2>", s1).lower()


def _collect_signature_params(signatures: list[dict]) -> set[str]:
    """Collect all param names from function signatures (incl. snake_case)."""
    names: set[str] = set()
    for sig in signatures:
        for p in sig.get("parameters", []):
            name = p.get("name", "") if isinstance(p, dict) else str(p)
            if name:
                names.add(name)
                names.add(_camel_to_snake(name))
    return names


def _extract_dim_vars(text: str) -> set[str]:
    """Extract named dimension variables from shape tuple strings.

    Skips parenthesized content that looks like a Markdown hyperlink URL
    (e.g. ``../common/非连续的Tensor.md``) to avoid extracting spurious
    identifiers such as ``common`` or ``md``.
    """
    dim_vars: set[str] = set()
    for match in _SHAPE_TUPLE_RE.finditer(text):
        tuple_content = match.group(1)
        if _is_markdown_url(tuple_content):
            continue
        for var_match in _DIM_VAR_RE.finditer(tuple_content):
            var_name = var_match.group(1)
            if var_name not in _EXCLUDE_WORDS:
                dim_vars.add(var_name)
    return dim_vars


def _find_nearby_param_name(text: str, pos: int) -> str:
    """Find the parameter name from the HTML table row containing pos.

    Looks backward for the nearest <tr tag, then extracts text from the
    first <td> cell in that row.
    """
    tr_start = text.rfind("<tr", 0, pos)
    if tr_start < 0:
        return ""
    td_start = text.find("<td", tr_start)
    if td_start < 0 or td_start > pos:
        return ""
    content_start = text.find(">", td_start) + 1
    td_end = text.find("</td", content_start)
    if td_end < 0:
        return ""
    cell = re.sub(r"<[^>]+>", "", text[content_start:td_end]).strip()
    return cell if cell.isidentifier() else ""


# ---------------------------------------------------------------------------
# Core identification logic
# ---------------------------------------------------------------------------


def _identify_implicit_params(
    sections_text: str,
    signatures: list[dict],
) -> list[dict]:
    """Identify implicit dimension variables from raw section text.

    Returns a list of parameter records ready for save_parameters.
    Uses field names expected by the MCP save_params tool:
    data_type (-> dtype_desc column), data_format (-> dformat_desc).
    """
    sig_params = _collect_signature_params(signatures)
    dim_var_usage: dict[str, set[str]] = {}

    for match in _SHAPE_TUPLE_RE.finditer(sections_text):
        tuple_content = match.group(1)
        # Skip Markdown hyperlink URLs like (../common/xxx.md)
        if _is_markdown_url(tuple_content):
            continue
        for var_match in _DIM_VAR_RE.finditer(tuple_content):
            var_name = var_match.group(1)
            if var_name in _EXCLUDE_WORDS or var_name in sig_params:
                continue
            dim_var_usage.setdefault(var_name, set())
            nearby = _find_nearby_param_name(sections_text, match.start())
            if nearby:
                dim_var_usage[var_name].add(nearby)

    if not dim_var_usage:
        return []

    # Determine GetWorkspaceSize function name
    ws_fn = next(
        (
            sig["function_name"]
            for sig in signatures
            if sig.get("function_name", "").endswith("GetWorkspaceSize")
        ),
        "",
    )
    if not ws_fn:
        logger.warning(
            "ImplicitParamExtract: no GetWorkspaceSize function in signatures"
        )
        return []

    implicit_params: list[dict] = []
    for var_name, affected in sorted(dim_var_usage.items()):
        logger.debug(
            "ImplicitParamExtract: found '%s' affecting: %s",
            var_name, sorted(affected),
        )
        implicit_params.append({
            "function_name": ws_fn,
            "param_name": var_name,
            "param_type": "int64_t",
            "direction": "input",
            "shape": "标量",
            "data_type": "INT",
            "data_format": "",
            "src_content": "",
        })

    return implicit_params


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


async def implicit_param_extract_node(state: PipelineState) -> dict[str, Any]:
    """Identify implicit dimension variables and inject into parameters table.

    Reads raw section content via MCP, scans for named dimension variables
    not present in any function signature, and writes them as new parameter
    records. Returns implicit_params in state for downstream consumption
    by param_relation_extract.

    Does NOT modify state["parameters"] -- see scheme section 3.3.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ImplicitParamExtract: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ImplicitParamExtract: no doc_id, skipping")
        return {"error": None}

    try:
        # Step 1: Fetch raw section text containing HTML parameter tables
        sections_text = ""
        for section_type in _SECTION_TYPES:
            sec = await _mcp_client.get_section(doc_id, section_type)
            if sec and sec.get("content"):
                sections_text += sec["content"] + "\n\n"

        if not sections_text.strip():
            logger.info(
                "ImplicitParamExtract: no section content for doc_id=%s", doc_id
            )
            return {"implicit_params": [], "error": None}

        # Step 2: Fetch function signatures
        sigs = await _mcp_client.query_function_signatures_by_doc_id(doc_id)

        # Step 3: Identify implicit dimension variables
        implicit_params = _identify_implicit_params(sections_text, sigs)

        if not implicit_params:
            logger.info(
                "ImplicitParamExtract: no implicit params found for %s",
                operator_name,
            )
            return {"implicit_params": [], "error": None}

        # Step 4: Inject into parameters table (INSERT OR REPLACE, idempotent)
        result = await _mcp_client.save_parameters(doc_id, implicit_params)
        logger.info(
            "ImplicitParamExtract: injected %d implicit params for %s: %s",
            result.get("saved", 0),
            operator_name,
            [p["param_name"] for p in implicit_params],
        )

        # Return via state for param_relation_extract
        # NOT added to state["parameters"] to avoid unnecessary LLM calls
        # in llm_description_extract downstream
        return {"implicit_params": implicit_params, "error": None}

    except Exception as e:
        logger.exception("ImplicitParamExtract failed for %s", operator_name)
        return {"error": str(e)}
