"""ImplicitParamExtract node: extract non-operator (implicit) parameters from document text.

Two-stage architecture (Plan A — "generate then filter"):

Phase 0 (deterministic): Identify tensor parameters from HTML table rows.
Phase 1 (deterministic): Regex coarse-sieve — collect ALL candidate named
    identifiers from shape tuples, including context (±100 chars) for the
    Agent to inspect.  No classification judgment is made here; false
    positives (e.g. "Reduce" in "Reduce维度") are expected and tolerated.
Phase 2 (LLM Agent):   Validate and classify each candidate.  The Agent
    confirms true dimension variables, removes concept terms / operation
    names, reclassifies constants and external constants, and can supplement
    missed parameters discovered in constraint text.
Phase 3 (deterministic): Extract platform-specific values for external
    constants (e.g. "Atlas A2：支持2、4、8卡").
Phase 4 (deterministic): Always inject the quantization_type implicit
    parameter (char-typed enum).

If the Agent fails (timeout, parse error, etc.), the system degrades
gracefully to the Phase 1 regex results — the pipeline never breaks.

Position in subgraph:
    fetch_sections -> **this node** -> [extract_ws || extract_exe]
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from agent.mcp_client import MCPClient
from agent.nodes.param_relation_extract.prompts import (
    format_implicit_params_context,
)
from agent.nodes.param_relation_extract.state import RelationExtractState
from agent.utils.llm_common import parse_json_response

logger = logging.getLogger(__name__)

_mcp_client = MCPClient()

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Match shape tuples: Chinese or ASCII parentheses
_SHAPE_TUPLE_RE = re.compile(r"[（(\[]\s*([^）)\]]+)\s*[）)\]]")

# Named dimension variable: any-case identifier
_DIM_VAR_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)*)\b")

# Words that look like dimension variables but are not.
# Used as a *coarse pre-filter* in Phase 1 — the Agent makes the final call.
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
    # -- C base type keywords --
    "float", "double", "void", "char", "int", "long", "short",
    "signed", "unsigned", "struct", "union", "enum",
    "Float", "Double", "Void", "Char", "Int", "Long", "Short",
    "Signed", "Unsigned", "Struct", "Union", "Enum",
    # -- C fixed-width type keywords --
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "ptrdiff_t",
    # -- Bare data-type names --
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
    # -- English stop-words --
    "or", "and", "if", "else", "when",
    "the", "for", "not", "with", "from",
    "Or", "And", "If", "Else", "When",
    "The", "For", "Not", "With", "From",
})

# ---------------------------------------------------------------------------
# Quantization type (default implicit parameter)
# ---------------------------------------------------------------------------

_QUANTIZATION_CANDIDATES = (
    "per-channel",
    "per-group",
    "per-tensor",
    "per-token",
)

_QUANTIZATION_PARAM_NAME = "quantization_type"

# ---------------------------------------------------------------------------
# Context window for surrounding_text (chars on each side of the match)
# ---------------------------------------------------------------------------

_CONTEXT_RADIUS = 100

# ---------------------------------------------------------------------------
# Deterministic helpers (retained from original implementation)
# ---------------------------------------------------------------------------


def _is_markdown_url(content: str) -> bool:
    """Return True if *content* looks like a Markdown hyperlink URL."""
    stripped = content.strip()
    if "/" in stripped:
        return True
    if stripped.endswith(".md") or stripped.endswith(".html"):
        return True
    if stripped.startswith("..") or stripped.startswith("./"):
        return True
    return False


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
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
    """Find the parameter name from the HTML table row containing *pos*."""
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
    if cell.isidentifier():
        return cell
    # Cell may contain a type annotation in parentheses, e.g.
    # "x（aclTensor*）" or "activation（char*）".  Extract the leading
    # identifier before the opening parenthesis.
    m = re.match(r"^([A-Za-z_]\w*)", cell)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Phase 0: Tensor parameter identification (deterministic)
# ---------------------------------------------------------------------------


def _identify_tensor_params(
    sections_text: str,
    sig_params: set[str],
) -> set[str]:
    """Collect names of tensor-type parameters from HTML table rows.

    A parameter is a tensor if its row has shape info or tensor data types.
    """
    tensor_params: set[str] = set()

    for row_match in re.finditer(
        r"<tr>\s*<td>([^<]*)</td>(.*?)</tr>",
        sections_text,
        re.DOTALL,
    ):
        # Cell may contain a type annotation in parentheses, e.g.
        # "x（aclTensor*）".  Extract the leading identifier.
        raw_name = row_match.group(1).strip()
        m = re.match(r"^([A-Za-z_]\w*)", raw_name)
        param_name = m.group(1) if m else raw_name
        if param_name not in sig_params:
            continue
        row_content = row_match.group(2)
        cells = re.findall(r"<td>(.*?)</td>", row_content, re.DOTALL)
        for cell in cells:
            clean = re.sub(r"<[^>]+>", "", cell).strip()
            if "shape" in clean.lower() and clean != "-":
                tensor_params.add(param_name)
                break
            if any(
                dt in clean
                for dt in ("FLOAT16", "BFLOAT16", "FLOAT32", "INT8", "INT32", "INT64")
            ):
                tensor_params.add(param_name)
                break

    return tensor_params


# ---------------------------------------------------------------------------
# Phase 1 helpers: slot-aware parsing (deterministic)
# ---------------------------------------------------------------------------


def _parse_shape_slots(tuple_content: str) -> list[dict]:
    """Parse a shape tuple into slot-aware dimension mappings.

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


def _should_skip_shape_tuple(
    nearby_param: str,
    tensor_params: set[str],
) -> bool:
    """Return True if this shape tuple should be skipped (non-tensor param)."""
    return nearby_param not in tensor_params

# ---------------------------------------------------------------------------
# Phase 1: Regex coarse-sieve candidate collection (deterministic)
# ---------------------------------------------------------------------------


def _collect_candidates(
    sections_text: str,
    signatures: list[dict],
    tensor_params: set[str] | None = None,
) -> list[dict]:
    """Collect ALL candidate named identifiers from shape tuples.

    This is the "generate" stage of generate-then-filter.  It deliberately
    has **high recall** and tolerates false positives (e.g. "Reduce" in
    "Reduce维度").  The Agent (Phase 2) will filter them out.

    Each candidate record includes:
    - candidate_id: unique identifier for Agent reference
    - var_name: the extracted identifier
    - tensor_param: the nearby tensor parameter name
    - dim_index / slot_index: dimension slot position
    - slot_expr / is_compound / compound_expr: slot metadata
    - shape_text: the original shape tuple
    - surrounding_text: ±100 chars of context around the shape tuple
    """
    sig_params = _collect_signature_params(signatures)
    _tensor_params = tensor_params or set()
    candidates: list[dict] = []
    cand_counter = 0

    for match in _SHAPE_TUPLE_RE.finditer(sections_text):
        tuple_content = match.group(1)
        if _is_markdown_url(tuple_content):
            continue

        nearby_param = _find_nearby_param_name(sections_text, match.start())
        if not nearby_param:
            continue

        if _tensor_params and _should_skip_shape_tuple(
            nearby_param, _tensor_params,
        ):
            logger.debug(
                "ImplicitParamExtract: skipping shape tuple in non-tensor "
                "param row '%s': (%s)",
                nearby_param, tuple_content,
            )
            continue

        # Extract ±100 chars of surrounding context for the Agent
        ctx_start = max(0, match.start() - _CONTEXT_RADIUS)
        ctx_end = min(len(sections_text), match.end() + _CONTEXT_RADIUS)
        surrounding = sections_text[ctx_start:ctx_end]

        slots = _parse_shape_slots(tuple_content)
        for slot in slots:
            for var in slot["vars"]:
                if var in sig_params:
                    continue
                cand_counter += 1
                candidates.append({
                    "candidate_id": f"cand_{cand_counter:03d}",
                    "var_name": var,
                    "tensor_param": nearby_param,
                    "dim_index": slot["slot_index"],
                    "slot_index": slot["slot_index"],
                    "slot_expr": slot["slot_expr"],
                    "is_compound": slot["is_compound"],
                    "compound_expr": (
                        slot["slot_expr"] if slot["is_compound"] else None
                    ),
                    "shape_text": f"({tuple_content.strip()})",
                    "surrounding_text": surrounding,
                })

    logger.info(
        "ImplicitParamExtract: Phase 1 collected %d candidates",
        len(candidates),
    )
    return candidates


# ---------------------------------------------------------------------------
# Phase 3: Platform constant value extraction (deterministic, retained)
# ---------------------------------------------------------------------------


def _extract_platform_constant_values(
    sections_text: str,
    const_name: str,
    supported_platforms: list[str],
) -> list[dict]:
    """Extract platform-specific values for an external constant.

    Scans constraint section for patterns like:
    - "Atlas A2 ...：支持2、4、8卡"

    ``<term>...</term>`` wrappers are stripped first so that platform names
    appear directly before the colon — the md wraps platform names in
    ``<term>`` tags, which would otherwise sit between the name and the colon
    and break the match (e.g. ``<term>Atlas A2</term>：``).
    """
    # Strip <term> wrappers so the platform name is directly followed by the colon.
    clean_text = re.sub(r"</?term>", "", sections_text)
    results: list[dict] = []
    for platform in supported_platforms:
        pattern = re.compile(
            re.escape(platform) + r"[：:].*?支持\s*([\d、,，\s]+)\s*卡",
        )
        match = pattern.search(clean_text)
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
# Phase 4: Quantization type extraction (deterministic, retained)
# ---------------------------------------------------------------------------


def _extract_quantization_modes(
    sections_text: str,
) -> tuple[list[str], str]:
    """Scan section text for quantization granularity modes."""
    matched: list[str] = []
    first_citation = ""
    for candidate in _QUANTIZATION_CANDIDATES:
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
    """Build the default quantization_type implicit parameter mapping."""
    modes, citation = _extract_quantization_modes(sections_text)
    return {
        "var_name": _QUANTIZATION_PARAM_NAME,
        "is_quantization_type": True,
        "param_type": "char",
        "allowed_range_value": modes,
        "allowed_range_type": "enum",
        "source_citation": citation,
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
# Phase 2: Agent validation and classification (LLM — core)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
)

_implicit_params_agent: Any = None


def _load_implicit_params_knowledge() -> str:
    """Eager-load all .md files under knowledge/implicit_params/."""
    kb_dir = os.path.join(_PROJECT_ROOT, "knowledge", "implicit_params")
    if not os.path.isdir(kb_dir):
        logger.warning("ImplicitParamsAgent: knowledge dir not found: %s", kb_dir)
        return ""
    parts: list[str] = []
    for root, _dirs, files in os.walk(kb_dir):
        for fname in sorted(files):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    parts.append(f.read().strip())
            except OSError:
                logger.warning("ImplicitParamsAgent: cannot read %s", fpath)
    return "\n\n---\n\n".join(parts)


_AGENT_SYSTEM_PROMPT_TEMPLATE = (
    "You are an implicit parameter validation expert for CANN operator documents.\n\n"
    "Your task: review candidate named identifiers that were regex-extracted from "
    "shape tuples, and judge whether each one is a real named dimension variable "
    "or a concept term / operation name that should be removed.\n\n"
    "## Your Input\n\n"
    "1. candidates: a JSON array of candidate records, each containing:\n"
    "   - candidate_id, var_name, tensor_param, dim_index\n"
    "   - slot_expr, is_compound, shape_text\n"
    "   - surrounding_text (plus/minus 100 chars of context)\n\n"
    "2. section_text: the full section text for context lookup\n"
    "3. signature_params: function signature parameter names (already excluded)\n"
    "4. tensor_params: tensor parameter names identified in Phase 0\n\n"
    "## Validation Rules\n\n"
    "### Rule 1: Named dimension variable (confirm)\n"
    "An identifier is a named dimension variable when:\n"
    "- It appears in a shape tuple representing a dimension **size value**\n"
    "- It is a symbolic variable name (e.g. N, C, H, W, BS, batchSize, k0)\n"
    "- It can be referenced in constraint expressions (e.g. BS.range_value)\n\n"
    "### Rule 2: Concept term (remove)\n"
    "Remove an identifier if it belongs to any of these categories:\n\n"
    "a) Dimension concept name: in X维度, X describes the dimension meaning,\n"
    "   not a variable. Examples: Reduce维度, GEMV维度, Attention维度.\n"
    "b) Operation/algorithm name: Conv, Softmax, ReLU, Sigmoid, GELU,\n"
    "   LayerNorm, BatchNorm, Matmul, Transpose, Reshape, etc.\n"
    "c) Data type name: float16, int32, bool, bfloat16, etc.\n"
    "d) Generic descriptor: shape, dtype, format, input, output, tensor, etc.\n\n"
    "### Rule 3: Constant (reclassify)\n"
    "If the text has an explicit assignment like k0 = 16, k0为16,\n"
    "k0等于16, mark as classification=constant, constant_value=number.\n\n"
    "### Rule 4: External constant (reclassify)\n"
    "If an identifier ONLY appears in compound expressions (e.g. H*rankSize)\n"
    "and never as a standalone dimension slot, mark as\n"
    "classification=external_constant. These typically depend on platform config.\n\n"
    "### Rule 5: Supplement missed parameters (additions)\n"
    "If you find a named dimension variable in section_text NOT captured\n"
    "by the regex, add it to the additions list.\n\n"
    "## Output Format\n"
    "Return ONLY a JSON object (no other text):\n"
    "{\n"
    '  "actions": [\n'
    '    {\n'
    '      "candidate_id": "cand_001",\n'
    '      "action": "confirm" | "remove" | "reclassify",\n'
    '      "classification": "dimension_variable" | "constant" | "external_constant",\n'
    '      "var_name": "BS",\n'
    '      "tensor_param": "x1",\n'
    '      "dim_index": 0,\n'
    '      "constant_value": null,\n'
    '      "referenced_in": [],\n'
    '      "reason": "why this judgment was made"\n'
    '    }\n'
    '  ],\n'
    '  "additions": [\n'
    '    {\n'
    '      "var_name": "rankSize",\n'
    '      "classification": "external_constant",\n'
    '      "tensor_param": null,\n'
    '      "dim_index": null,\n'
    '      "constant_value": null,\n'
    '      "referenced_in": ["x1"],\n'
    '      "reason": "found in constraint text"\n'
    '    }\n'
    '  ]\n'
    "}\n"
)


def _get_implicit_params_agent() -> Any:
    """Lazily create and cache the LLM for implicit param validation."""
    global _implicit_params_agent
    if _implicit_params_agent is not None:
        return _implicit_params_agent

    from agent.utils.llm_common import create_llm

    kb = _load_implicit_params_knowledge()
    system_prompt = _AGENT_SYSTEM_PROMPT_TEMPLATE
    if kb:
        system_prompt = system_prompt + "\n\n## Knowledge Base\n\n" + kb

    _implicit_params_agent = create_llm()
    _implicit_params_agent._implicit_params_system_prompt = system_prompt
    logger.info("ImplicitParamsAgent: created (KB=%d chars)", len(kb))
    return _implicit_params_agent


def _build_agent_user_message(
    candidates: list[dict],
    section_text: str,
    sig_params: set[str],
    tensor_params: set[str],
) -> str:
    """Build the user message for the Agent with all context."""
    return json.dumps({
        "candidates": candidates,
        "section_text": section_text,
        "signature_params": sorted(sig_params),
        "tensor_params": sorted(tensor_params),
    }, ensure_ascii=False)


def _parse_agent_response(text: str) -> dict[str, list] | None:
    """Parse the Agent JSON response into actions and additions.

    Returns None if parsing fails (triggers fallback).
    """
    data = parse_json_response(text, dict)
    if data is not None:
        actions = data.get("actions", [])
        additions = data.get("additions", [])
        if isinstance(actions, list) and isinstance(additions, list):
            return {"actions": actions, "additions": additions}
    logger.warning("ImplicitParamsAgent: failed to parse response: %s", text[:200])
    return None


def _candidate_to_mapping(candidate: dict) -> dict:
    """Convert a candidate record to the final mapping structure."""
    return {
        "var_name": candidate["var_name"],
        "tensor_param": candidate["tensor_param"],
        "dim_index": candidate["dim_index"],
        "shape_text": candidate["shape_text"],
        "is_constant": False,
        "constant_value": None,
        "slot_index": candidate["slot_index"],
        "slot_expr": candidate["slot_expr"],
        "is_compound": candidate["is_compound"],
        "compound_expr": candidate.get("compound_expr"),
        "is_external_constant": False,
        "is_quantization_type": False,
        "referenced_in": [],
    }


def _apply_agent_actions(
    candidates: list[dict],
    parsed: dict[str, list],
) -> list[dict]:
    """Apply Agent actions (confirm/remove/reclassify) and additions to candidates.

    Returns the final list of mapping dicts.
    """
    cand_by_id = {c["candidate_id"]: c for c in candidates}
    handled_ids: set[str] = set()
    mappings: list[dict] = []

    for action in parsed.get("actions", []):
        cid = action.get("candidate_id", "")
        act = action.get("action", "")
        reason = action.get("reason", "")

        if cid not in cand_by_id:
            logger.warning("ImplicitParamsAgent: unknown candidate_id %s", cid)
            continue

        handled_ids.add(cid)
        candidate = cand_by_id[cid]

        if act == "remove":
            logger.debug(
                "ImplicitParamsAgent: removed %s - %s",
                candidate["var_name"], reason,
            )
            continue

        if act in ("confirm", "reclassify"):
            mapping = _candidate_to_mapping(candidate)
            classification = action.get("classification", "dimension_variable")

            if classification == "constant":
                mapping["is_constant"] = True
                mapping["constant_value"] = action.get("constant_value")
            elif classification == "external_constant":
                mapping["is_external_constant"] = True
                mapping["tensor_param"] = None
                mapping["dim_index"] = None
                mapping["shape_text"] = None
                mapping["slot_index"] = None
                mapping["slot_expr"] = None
                mapping["is_compound"] = False
                mapping["compound_expr"] = None
                mapping["referenced_in"] = action.get("referenced_in", [])

            mappings.append(mapping)
            logger.debug(
                "ImplicitParamsAgent: %s %s as %s - %s",
                act, candidate["var_name"], classification, reason,
            )
            continue

        logger.warning(
            "ImplicitParamsAgent: unknown action %s for %s",
            act, candidate["var_name"],
        )

    # Add additions (missed parameters discovered by Agent)
    for addition in parsed.get("additions", []):
        var = addition.get("var_name", "")
        if not var:
            continue
        classification = addition.get("classification", "dimension_variable")
        mapping = {
            "var_name": var,
            "tensor_param": addition.get("tensor_param"),
            "dim_index": addition.get("dim_index"),
            "shape_text": None,
            "is_constant": classification == "constant",
            "constant_value": addition.get("constant_value"),
            "slot_index": None,
            "slot_expr": None,
            "is_compound": False,
            "compound_expr": None,
            "is_external_constant": classification == "external_constant",
            "is_quantization_type": False,
            "referenced_in": addition.get("referenced_in", []),
        }
        mappings.append(mapping)
        logger.debug(
            "ImplicitParamsAgent: added %s as %s - %s",
            var, classification, addition.get("reason", ""),
        )

    # Degradation: keep unhandled candidates (Agent missed them)
    unhandled = [c for c in candidates if c["candidate_id"] not in handled_ids]
    if unhandled:
        logger.warning(
            "ImplicitParamsAgent: %d candidates not handled, keeping as-is",
            len(unhandled),
        )
        for c in unhandled:
            mappings.append(_candidate_to_mapping(c))

    return mappings


async def _validate_via_agent(
    candidates: list[dict],
    section_text: str,
    sig_params: set[str],
    tensor_params: set[str],
) -> list[dict]:
    """Phase 2: Call the Agent to validate and classify candidates.

    Returns the final list of mapping dicts.
    Falls back to regex candidates on any failure.
    """
    if not candidates:
        return []

    agent = _get_implicit_params_agent()
    user_msg = _build_agent_user_message(
        candidates, section_text, sig_params, tensor_params,
    )

    system_prompt = getattr(agent, "_implicit_params_system_prompt", "")
    full_prompt = system_prompt + "\n\n## Input Data\n\n" + user_msg

    try:
        response = await agent.ainvoke(full_prompt)
        ai_text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.warning(
            "ImplicitParamsAgent: invocation failed, falling back to regex results",
            exc_info=True,
        )
        return [_candidate_to_mapping(c) for c in candidates]

    parsed = _parse_agent_response(ai_text)
    if parsed is None:
        logger.warning(
            "ImplicitParamsAgent: unparseable response, falling back to regex results",
        )
        return [_candidate_to_mapping(c) for c in candidates]

    mappings = _apply_agent_actions(candidates, parsed)

    # Degradation 3: too few results — merge with regex fallback
    if len(mappings) < len(candidates) * 0.3:
        logger.warning(
            "ImplicitParamsAgent: too few results (%d/%d), merging with fallback",
            len(mappings), len(candidates),
        )
        existing_vars = {m["var_name"] for m in mappings}
        for c in candidates:
            if c["var_name"] not in existing_vars:
                mappings.append(_candidate_to_mapping(c))

    logger.info(
        "ImplicitParamsAgent: validated %d candidates -> %d mappings",
        len(candidates), len(mappings),
    )
    return mappings


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


async def implicit_param_extract_node(
    state: RelationExtractState,
) -> dict[str, Any]:
    """Extract implicit (non-operator) parameters from document sections.

    Two-stage architecture (Plan A — generate then filter):
    - Phase 0 (deterministic): Identify tensor parameters
    - Phase 1 (deterministic): Regex coarse-sieve — collect candidates
    - Phase 2 (LLM Agent): Validate and classify candidates
    - Phase 3 (deterministic): Extract platform constant values
    - Phase 4 (deterministic): Inject quantization_type parameter

    If the Agent fails, the system degrades gracefully to Phase 1 regex
    results — the pipeline never breaks.
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

        # Phase 0: Identify tensor parameters (deterministic)
        tensor_params = _identify_tensor_params(sections_text, sig_params)
        logger.debug(
            "ImplicitParamExtract: Phase 0 identified %d tensor params: %s",
            len(tensor_params), sorted(tensor_params),
        )

        # Phase 1: Regex coarse-sieve — collect candidates (deterministic)
        candidates = _collect_candidates(
            sections_text, sigs, tensor_params,
        )

        # Phase 2: Agent validation and classification (LLM)
        if candidates:
            mappings = await _validate_via_agent(
                candidates, sections_text, sig_params, tensor_params,
            )
        else:
            mappings = []

        # Phase 4: Always inject the default quantization_type implicit param.
        quant_mapping = _build_quantization_type_mapping(sections_text)
        mappings.append(quant_mapping)

        # Phase 3: Extract platform constant values for external constants
        platforms = await _mcp_client.query_platform_support_by_doc_id(doc_id)
        supported = [
            p["platform_name"]
            for p in platforms if p.get("is_supported") == 1
        ]

        platform_constants: list[dict] = []
        seen_consts: set[str] = set()
        for m in mappings:
            if not m.get("is_external_constant"):
                continue
            cname = m["var_name"]
            # Dedup by const_name: the same external constant (e.g. rankSize)
            # may be classified multiple times across different text contexts;
            # keep only the first occurrence to avoid duplicate injected entries.
            if cname in seen_consts:
                continue
            seen_consts.add(cname)
            pv = _extract_platform_constant_values(
                sections_text, cname, supported,
            )
            if pv:
                platform_constants.append({
                    "const_name": cname,
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
        logger.exception("ImplicitParamExtract failed for %s", operator_name)
        return {"error": str(e)}
