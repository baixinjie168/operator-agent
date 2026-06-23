"""ImplicitParamExtract node: extract non-operator (implicit) parameters from document text.

Extracts named dimension variables (e.g. BS, H, N, b, m, k, n1) from
shape tuples in parameter tables.  These named variables are treated as
implicit (non-operator) parameters: they appear in constraint expressions
by name, not substituted with tensor.shape[i].

Supports:
- Slot-aware parsing for compound expressions (e.g. H*rankSize, BS/rankSize)
- Non-tensor parameter row filtering (e.g. bool transposeX2)
- External constant detection (e.g. rankSize = NPU card count)
- Platform constant value extraction from constraint sections

For example, for aclnnAlltoAllMatmul:
  BS -> x1.shape[0]
  H  -> x1.shape[1]
  N  -> x2.shape[1] | biasOptional.shape[0]
  rankSize -> external constant (platform-dependent values)

The result is persisted to DB (implicit_params table) for traceability,
and passed to downstream nodes via state["implicit_params"].

Zero LLM calls - purely deterministic regex-based extraction.

Position in subgraph:
    fetch_sections -> **this node** -> [extract_ws || extract_exe]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.prompts import (
    format_implicit_params_context,
)
from agent.nodes.param_relation_extract.state import RelationExtractState

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Match shape tuples: Chinese or ASCII parentheses
_SHAPE_TUPLE_RE = re.compile(r"[（(]\s*([^)）]+)\s*[）)]")

# Named dimension variable: any-case identifier (supports BS, H, N, rankSize, b, m, k, etc.)
_DIM_VAR_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)*)\b")

# Words that look like dimension variables but are not.
#
# Organised by category.  When adding new entries, keep them sorted
# within each group so duplicates are easy to spot.
_EXCLUDE_WORDS = frozenset({
    # -- Python / JSON literals --
    "true", "false", "none", "null",
    "True", "False", "None",

    # -- Common non-dimension terms --
    "shape", "dtype", "format", "type",
    "input", "output", "tensor", "optional",
    "nd", "acl",
    "Shape", "Dtype", "Format", "Type",
    "Input", "Output", "Tensor", "Optional",
    "ND", "ACL",

    # -- C base type keywords (no digit suffix) --
    "float", "double", "void", "char", "int", "long", "short",
    "signed", "unsigned", "struct", "union", "enum",
    "Float", "Double", "Void", "Char", "Int", "Long", "Short",
    "Signed", "Unsigned", "Struct", "Union", "Enum",

    # -- C fixed-width type keywords (with _t suffix) --
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "ptrdiff_t",

    # -- Bare data-type names (with digit suffix, already existed) --
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool", "string",
    "FLOAT16", "FLOAT32", "FLOAT64",
    "INT8", "INT16", "INT32", "INT64",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "BOOL", "STRING",

    # -- Common non-variable identifiers from Markdown / URLs --
    "common", "md", "html", "http", "https", "www",
    "aclnn", "aclrt", "device", "host",
    "Common", "Md", "Html", "Http", "Https", "Www",
    "Aclnn", "Aclrt", "Device", "Host",

    # -- English stop-words / logical operators --
    "or", "and", "if", "else", "when",
    "the", "for", "not", "with", "from",
    "Or", "And", "If", "Else", "When",
    "The", "For", "Not", "With", "From",
})

# ---------------------------------------------------------------------------
# Quantization type (default implicit parameter)
# ---------------------------------------------------------------------------

# The fixed universe of quantization granularity modes.  The parameter's
# allowed_range_value is the subset of these that actually appear in the
# operator's section text (preserving this canonical order, de-duplicated).
_QUANTIZATION_CANDIDATES = (
    "per-channel",
    "per-group",
    "per-tensor",
    "per-token",
)

_QUANTIZATION_PARAM_NAME = "quantization_type"


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
# Phase 0: Tensor parameter identification
# ---------------------------------------------------------------------------


def _identify_tensor_params(
    sections_text: str,
    sig_params: set[str],
) -> set[str]:
    """Collect names of tensor-type parameters from HTML table rows.

    A parameter is considered a tensor if:
    1. Its row contains a non-dash shape column value (e.g. "2维，shape为(BS, H)")
    2. Or its type column contains tensor data types (FLOAT16, BFLOAT16, etc.)
    """
    tensor_params: set[str] = set()

    # Scan HTML table rows: <tr><td>param_name</td>...<td>shape_info</td>...
    for row_match in re.finditer(
        r"<tr>\s*<td>(\w+)</td>(.*?)</tr>",
        sections_text,
        re.DOTALL,
    ):
        param_name = row_match.group(1)
        if param_name not in sig_params:
            continue
        row_content = row_match.group(2)
        cells = re.findall(r"<td>(.*?)</td>", row_content, re.DOTALL)
        for cell in cells:
            clean = re.sub(r"<[^>]+>", "", cell).strip()
            # Has shape info (not just "-")
            if "shape" in clean.lower() and clean != "-":
                tensor_params.add(param_name)
                break
            # Has tensor data types
            if any(
                dt in clean
                for dt in ("FLOAT16", "BFLOAT16", "FLOAT32", "INT8", "INT32", "INT64")
            ):
                tensor_params.add(param_name)
                break

    return tensor_params


# ---------------------------------------------------------------------------
# Phase 1: Slot-aware dimension parsing
# ---------------------------------------------------------------------------


def _parse_shape_slots(tuple_content: str) -> list[dict]:
    """Parse a shape tuple into slot-aware dimension mappings.

    Splits by comma first to identify dimension slots, then extracts
    named variables within each slot.

    Example: "H*rankSize, N" returns:
    [
        {"slot_index": 0, "slot_expr": "H*rankSize",
         "vars": ["H", "rankSize"], "is_compound": True},
        {"slot_index": 1, "slot_expr": "N",
         "vars": ["N"], "is_compound": False},
    ]
    """
    slots = [s.strip() for s in tuple_content.split(",")]
    result: list[dict] = []
    for i, slot in enumerate(slots):
        dim_vars = [
            v
            for v in _DIM_VAR_RE.findall(slot)
            if v not in _EXCLUDE_WORDS
        ]
        has_op = bool(re.search(r"[*+\-/]", slot))
        result.append({
            "slot_index": i,
            "slot_expr": slot,
            "vars": dim_vars,
            "is_compound": has_op and len(dim_vars) > 1,
        })
    return result


# ---------------------------------------------------------------------------
# Phase 2: Non-tensor parameter row filtering
# ---------------------------------------------------------------------------


def _should_skip_shape_tuple(
    nearby_param: str,
    tensor_params: set[str],
) -> bool:
    """Return True if this shape tuple should be skipped.

    Skip when the nearby parameter is NOT a tensor type.
    For example, transposeX2 is a bool parameter whose description
    mentions "(N, rankSize*H)" — this shape describes x2's conditional
    form, not transposeX2's own shape.
    """
    return nearby_param not in tensor_params


# ---------------------------------------------------------------------------
# Phase 3: External constant detection
# ---------------------------------------------------------------------------


def _detect_external_constants(
    mappings: list[dict],
    sections_text: str,
) -> list[dict]:
    """Identify variables that are external constants, not tensor dimensions.

    A variable is an external constant if:
    1. It ONLY appears in compound slots (never as a standalone dimension)
    2. It is not already marked as a known constant (e.g. k0=16)

    Returns a list of external constant mapping records.
    """
    # Collect variables that have standalone (non-compound) mappings
    standalone_vars: set[str] = set()
    compound_vars: set[str] = set()

    for m in mappings:
        if m.get("is_constant"):
            continue
        if not m.get("is_compound"):
            standalone_vars.add(m["var_name"])
        else:
            compound_vars.add(m["var_name"])

    # Candidates: only in compound slots, never standalone
    candidates = compound_vars - standalone_vars

    external_constants: list[dict] = []
    for var in sorted(candidates):
        # Collect which params reference this constant
        refs = sorted({
            m["tensor_param"]
            for m in mappings
            if m["var_name"] == var and m.get("tensor_param")
        })

        # Optional: verify with section text for explicit definition
        is_confirmed = bool(re.search(
            r"(?:NPU|GPU|卡数|设备数|并行度)[（(]" + re.escape(var) + r"[）)]",
            sections_text,
        ))

        if is_confirmed or refs:
            external_constants.append({
                "var_name": var,
                "is_external_constant": True,
                "tensor_param": None,
                "dim_index": None,
                "shape_text": None,
                "is_constant": False,
                "constant_value": None,
                "slot_index": None,
                "slot_expr": None,
                "is_compound": False,
                "compound_expr": None,
                "referenced_in": refs,
            })
            logger.debug(
                "ImplicitParamExtract: detected external constant '%s' "
                "(referenced_in=%s, confirmed_by_text=%s)",
                var, refs, is_confirmed,
            )

    return external_constants


# ---------------------------------------------------------------------------
# Phase 4: Platform constant value extraction
# ---------------------------------------------------------------------------


def _extract_platform_constant_values(
    sections_text: str,
    const_name: str,
    supported_platforms: list[str],
) -> list[dict]:
    """Extract platform-specific values for an external constant.

    Scans constraint section for patterns like:
    - "Atlas A2 ...：支持2、4、8卡"
    - "Atlas 350 加速卡：支持2、4、8、16卡"

    Returns a list of {"platform", "values", "source_citation"} dicts.
    """
    results: list[dict] = []
    for platform in supported_platforms:
        # Pattern: platform name ... 支持 ... digit list ... 卡
        pattern = re.compile(
            re.escape(platform) + r"[：:].*?支持\s*([\d、,，\s]+)\s*卡",
        )
        match = pattern.search(sections_text)
        if match:
            values_str = match.group(1)
            values = [
                int(v.strip())
                for v in re.split(r"[、,，\s]+", values_str)
                if v.strip().isdigit()
            ]
            if values:
                results.append({
                    "platform": platform,
                    "values": sorted(values),
                    "source_citation": match.group(0).strip(),
                })
    return results


# ---------------------------------------------------------------------------
# Constant detection (existing, enhanced)
# ---------------------------------------------------------------------------


def _detect_constants(
    sections_text: str,
    mappings: list[dict],
) -> None:
    """Post-process mappings to detect constant dimension variables.

    Scans the section text for patterns like "其中k0 = 16" or "n0为16"
    and marks the corresponding mapping entries as constants.

    Modifies *mappings* in place.
    """
    if not mappings:
        return

    # Collect all var names that need checking
    var_names = {m["var_name"] for m in mappings if not m.get("is_constant")}
    if not var_names:
        return

    for var in var_names:
        # Pattern: "k0 = 16", "k0为16", "k0 等于 16", "k0 is 16"
        pattern = re.compile(
            r"\b" + re.escape(var) + r"\s*(?:[=为]|等于|is)\s*(\d+)",
            re.IGNORECASE,
        )
        match = pattern.search(sections_text)
        if match:
            const_val = int(match.group(1))
            for m in mappings:
                if m["var_name"] == var:
                    m["is_constant"] = True
                    m["constant_value"] = const_val
            logger.debug(
                "ImplicitParamExtract: detected constant %s = %d", var, const_val,
            )


# ---------------------------------------------------------------------------
# Shape dimension mapping extraction
# ---------------------------------------------------------------------------


def _build_implicit_params(
    sections_text: str,
    signatures: list[dict],
    tensor_params: set[str] | None = None,
) -> list[dict]:
    """Build shape dimension mappings from section text.

    Uses slot-aware parsing to correctly handle compound expressions
    (e.g. H*rankSize, BS/rankSize) and filters out non-tensor parameter rows.

    Returns a list of mapping dicts, each containing:
    - var_name: the named dimension variable (e.g. "b", "n1")
    - tensor_param: the tensor parameter it belongs to (e.g. "self", "mat2")
    - dim_index: the index in the tensor's shape tuple (= slot_index)
    - slot_index: comma-delimited slot position
    - slot_expr: the full expression text of this slot (e.g. "H*rankSize")
    - is_compound: whether this slot contains arithmetic operators
    - compound_expr: the compound expression (None if not compound)
    - shape_text: the original shape tuple text
    - is_constant: whether this is a known constant (e.g. k0=16)
    - constant_value: the constant value (only when is_constant=True)
    """
    sig_params = _collect_signature_params(signatures)
    _tensor_params = tensor_params or set()
    mappings: list[dict] = []

    for match in _SHAPE_TUPLE_RE.finditer(sections_text):
        tuple_content = match.group(1)
        # Skip Markdown hyperlink URLs like (../common/xxx.md)
        if _is_markdown_url(tuple_content):
            continue

        # Find the tensor parameter name this shape tuple belongs to
        nearby_param = _find_nearby_param_name(sections_text, match.start())
        if not nearby_param:
            continue

        # Phase 2: Skip shape tuples in non-tensor parameter rows
        if _tensor_params and _should_skip_shape_tuple(
            nearby_param, _tensor_params,
        ):
            logger.debug(
                "ImplicitParamExtract: skipping shape tuple in non-tensor "
                "param row '%s': (%s)",
                nearby_param, tuple_content,
            )
            continue

        # Phase 1: Slot-aware parsing
        slots = _parse_shape_slots(tuple_content)
        for slot in slots:
            for var in slot["vars"]:
                if var in sig_params:
                    continue
                mappings.append({
                    "var_name": var,
                    "tensor_param": nearby_param,
                    "dim_index": slot["slot_index"],
                    "shape_text": f"({tuple_content.strip()})",
                    "is_constant": False,
                    "constant_value": None,
                    "slot_index": slot["slot_index"],
                    "slot_expr": slot["slot_expr"],
                    "is_compound": slot["is_compound"],
                    "compound_expr": (
                        slot["slot_expr"] if slot["is_compound"] else None
                    ),
                })

    # Detect constants (e.g. k0=16, n0=16)
    _detect_constants(sections_text, mappings)

    return mappings

# ---------------------------------------------------------------------------
# Phase 5: Quantization type extraction (default implicit parameter)
# ---------------------------------------------------------------------------


def _extract_quantization_modes(
    sections_text: str,
) -> tuple[list[str], str]:
    """Scan section text for quantization granularity modes.

    Searches for each candidate in :data:`_QUANTIZATION_CANDIDATES` using
    word-boundary matching (case-sensitive) and returns those that appear,
    preserving the canonical candidate order and de-duplicating.

    Returns ``(matched_modes, source_citation)`` where *source_citation* is
    a short snippet around the first match (empty when nothing matches).
    """
    matched: list[str] = []
    first_citation = ""
    for candidate in _QUANTIZATION_CANDIDATES:
        # Use ASCII-letter lookbehind/lookahead instead of \b: Python's \b
        # treats Chinese characters as word chars (Unicode \w), which would
        # prevent matching candidates adjacent to Chinese text (e.g.
        # "支持per-channel" or "per-channel下").  The lookbehind still rejects
        # false positives like "super-channel" → "per-channel".
        pattern = re.compile(
            r"(?<![A-Za-z])" + re.escape(candidate) + r"(?![A-Za-z])"
        )
        m = pattern.search(sections_text)
        if m:
            matched.append(candidate)
            if not first_citation:
                start = max(0, m.start() - 20)
                end = min(len(sections_text), m.end() + 20)
                first_citation = sections_text[start:end].strip()
    return matched, first_citation


def _build_quantization_type_mapping(
    sections_text: str,
) -> dict[str, Any]:
    """Build the default ``quantization_type`` implicit parameter mapping.

    Always returns a mapping (even when no modes are found — in which case
    ``allowed_range_value`` is an empty list).  The parameter is a
    ``char``-typed enum constrained to the quantization granularity modes
    mentioned in the document.
    """
    modes, citation = _extract_quantization_modes(sections_text)
    return {
        "var_name": _QUANTIZATION_PARAM_NAME,
        "is_quantization_type": True,
        "param_type": "char",
        "allowed_range_value": modes,
        "allowed_range_type": "enum",
        "source_citation": citation,
        # Compatibility fields (kept consistent with shape-dim mappings so
        # that downstream consumers can treat it uniformly).
        "tensor_param": None,
        "dim_index": None,
        "shape_text": None,
        "slot_index": None,
        "slot_expr": None,
        "is_compound": False,
        "compound_expr": None,
        "is_constant": False,
        "constant_value": None,
        "is_external_constant": False,
        "referenced_in": [],
    }

# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


async def implicit_param_extract_node(
    state: RelationExtractState,
) -> dict[str, Any]:
    """Build shape dimension mappings and persist to DB for traceability.

    Reads section content from subgraph state (already fetched by
    fetch_sections_node), scans for named dimension variables not present
    in any function signature, and builds a mapping table.

    Enhanced with:
    - Phase 0: Tensor parameter identification
    - Phase 1: Slot-aware dimension parsing (compound expressions)
    - Phase 2: Non-tensor parameter row filtering
    - Phase 3: External constant detection
    - Phase 4: Platform constant value extraction

    Does NOT inject synthetic parameters into the parameters table.
    """
    doc_id = state.get("doc_id", 0)
    operator_name = state.get("operator_name", "")

    logger.info("ImplicitParamExtract: doc_id=%s for %s", doc_id, operator_name)

    if not doc_id:
        logger.warning("ImplicitParamExtract: no doc_id, skipping")
        return {
            "implicit_params": [],
            "platform_constants": [],
            "error": None,
        }

    try:
        # Combine ws + exe section content (already fetched by fetch_sections)
        ws_content = state.get("ws_section_content", "")
        exe_content = state.get("exe_section_content", "")
        sections_text = f"{ws_content}\n\n{exe_content}"

        if not sections_text.strip():
            logger.info(
                "ImplicitParamExtract: no section content for doc_id=%s", doc_id
            )
            return {
                "implicit_params": [],
                "platform_constants": [],
                "error": None,
            }

        # Fetch function signatures
        sigs = await _mcp_client.query_function_signatures_by_doc_id(doc_id)
        sig_params = _collect_signature_params(sigs)

        # Phase 0: Identify tensor parameters
        tensor_params = _identify_tensor_params(sections_text, sig_params)
        logger.debug(
            "ImplicitParamExtract: identified %d tensor params: %s",
            len(tensor_params), sorted(tensor_params),
        )

        # Phase 1+2: Build mappings with slot-aware parsing + tensor filtering
        mappings = _build_implicit_params(
            sections_text, sigs, tensor_params,
        )

        # Phase 5: Always inject the default quantization_type implicit param.
        # It is a char-typed enum whose allowed_range_value is the subset of
        # _QUANTIZATION_CANDIDATES that appear in the document (possibly empty).
        quant_mapping = _build_quantization_type_mapping(sections_text)
        mappings.append(quant_mapping)

        # Phase 3: Detect external constants
        ext_constants = _detect_external_constants(mappings, sections_text)
        mappings.extend(ext_constants)

        # Phase 4: Extract platform constant values
        platforms = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        supported = [
            p["platform_name"]
            for p in platforms if p.get("is_supported") == 1
        ]

        platform_constants: list[dict] = []
        for ec in ext_constants:
            pv = _extract_platform_constant_values(
                sections_text, ec["var_name"], supported,
            )
            if pv:
                platform_constants.append({
                    "const_name": ec["var_name"],
                    "description": "",
                    "platform_values": pv,
                })

        # Persist implicit_params to DB
        rendered = format_implicit_params_context(
            mappings, platform_constants,
        )
        await _mcp_client.save_implicit_params(
            doc_id=doc_id,
            mappings_json=json.dumps(mappings, ensure_ascii=False),
            rendered_text=rendered,
        )

        # Persist platform_constants to DB (if MCP tool available)
        if platform_constants:
            try:
                await _mcp_client.save_platform_constants(
                    doc_id=doc_id,
                    constants_json=json.dumps(
                        platform_constants, ensure_ascii=False,
                    ),
                )
            except Exception:
                logger.debug(
                    "ImplicitParamExtract: save_platform_constants not yet "
                    "available, skipping DB persist",
                )

        # Log summary
        var_names = sorted({
            m["var_name"] for m in mappings
            if not m.get("is_external_constant")
            and not m.get("is_quantization_type")
        })
        const_names = sorted({
            m["var_name"] for m in mappings if m.get("is_constant")
        })
        ext_const_names = sorted({
            m["var_name"] for m in mappings
            if m.get("is_external_constant")
        })
        compound_count = sum(
            1 for m in mappings if m.get("is_compound")
        )
        quant_modes = quant_mapping.get("allowed_range_value", [])
        logger.info(
            "ImplicitParamExtract: built %d mappings for %s: "
            "vars=%s, constants=%s, external=%s, compounds=%d, "
            "platform_constants=%d, quantization_type=%s",
            len(mappings),
            operator_name,
            var_names,
            const_names,
            ext_const_names,
            compound_count,
            len(platform_constants),
            quant_modes,
        )

        # Return via state for downstream nodes
        return {
            "implicit_params": mappings,
            "platform_constants": platform_constants,
            "error": None,
        }

    except Exception as e:
        logger.exception("ShapeDimMapping failed for %s", operator_name)
        return {"error": str(e)}
