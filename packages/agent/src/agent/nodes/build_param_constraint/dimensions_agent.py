"""DimensionsAgent node: parse shape text into structured dimensions arrays.

Flow:
1. Collect all unique shape values from params (same as dimensions_build)
2. Phase 1:   Deterministic regex parsing (zero LLM, ~70-80% coverage)
3. Phase 1.5: HTML-list shape parsing (zero LLM, handles <ul><li> variants)
4. Phase 2:   DeepAgent processes remaining shapes (knowledge base inlined
              into the system prompt via eager loading — no read_file round-trip)
5. Phase 3:   Structure validation

Fallback: if DeepAgent fails, returns empty dimensions for unparseable shapes.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from agent.nodes.build_param_constraint._helpers import _parse_json_field
from agent.nodes.build_param_constraint.state import BuildParamConstraintState
from agent.utils.llm_common import parse_json_response

logger = logging.getLogger(__name__)

# Lazy-loaded DeepAgent instance (created on first call)
_dimensions_agent: Any = None

# Project root for knowledge base path
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
)


# ---------------------------------------------------------------------------
# Knowledge base eager-loader
# ---------------------------------------------------------------------------

def _load_dimensions_knowledge() -> str:
    """Read all ``.md`` files under ``knowledge/dimensions/`` at agent creation.

    The content is appended to the system prompt so the LLM sees every rule
    and example without needing a ``read_file`` tool-call round-trip.

    Adding a new special-case rule = dropping a new ``.md`` file into
    ``knowledge/dimensions/examples/`` — no code change required.
    """
    kb_dir = os.path.join(_PROJECT_ROOT, "knowledge", "dimensions")
    if not os.path.isdir(kb_dir):
        logger.warning("DimensionsAgent: knowledge dir not found: %s", kb_dir)
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
                logger.warning("DimensionsAgent: cannot read %s", fpath)
    return "\n\n---\n\n".join(parts)


def _get_dimensions_agent() -> Any:
    """Lazily create and cache the DeepAgent for dimensions generation.

    The knowledge base (SKILL.md + examples/*.md) is **eager-loaded** into
    the system prompt rather than relying on the DeepAgent skill mechanism
    (progressive disclosure via ``read_file``).  This avoids an extra LLM
    round-trip per invocation and makes rule application deterministic.
    """
    global _dimensions_agent
    if _dimensions_agent is not None:
        return _dimensions_agent

    from deepagents import create_deep_agent
    from agent.core.llm import create_llm

    kb = _load_dimensions_knowledge()
    system_prompt = _DIMENSIONS_AGENT_SYSTEM_PROMPT
    if kb:
        system_prompt = system_prompt + "\n\n## Knowledge Base\n\n" + kb

    _dimensions_agent = create_deep_agent(
        model=create_llm(),
        tools=[],
        system_prompt=system_prompt,
        name="dimensions-generate-agent",
    )
    logger.info(
        "DimensionsAgent: created (KB=%d chars, %d rules)",
        len(kb), kb.count("\n## ") + kb.count("\n### "),
    )
    return _dimensions_agent


# ---------------------------------------------------------------------------
# System prompt for the DeepAgent
# ---------------------------------------------------------------------------
# Only the output-format contract lives here (critical — the calling code
# _parse_dimensions_response depends on this exact JSON shape).  All parsing
# RULES and EXAMPLES are in the knowledge base (knowledge/dimensions/) and
# get appended at agent creation time via _load_dimensions_knowledge().

_DIMENSIONS_AGENT_SYSTEM_PROMPT = """\
You are a dimensions parsing expert for CANN operator documents.

Convert shape descriptions into structured dimensions arrays.

## Output Format (MUST follow exactly)

Return a JSON array, one element per input shape, in order. Each element is:
- Rank format: [min_rank, max_rank]  (e.g. [4, 4] = exactly 4 dimensions)
- Per-dimension format: [[min,max], ...]  (e.g. [[2,2],[3,3],[4,4]])
- Empty array []  (scalar / cannot determine)

Return ONLY the JSON array, no explanation.

Apply the parsing rules and examples from the Knowledge Base below.
"""


# ---------------------------------------------------------------------------
# Phase 1: Deterministic preprocessing (reused from dimensions_build)
# ---------------------------------------------------------------------------

_DIMENSION_PATTERNS: list[tuple[str, Any]] = [
    (r"^(标量|0[- ]?D|0D)$", []),
    (
        r"^(\d+)\s*[-~]\s*(\d+)$",
        lambda m: [int(m.group(1)), int(m.group(2))],
    ),
    (r"^1[- ]?D$", [1, 1]),
    (r"^(\d+)[- ]?D$", lambda m: [int(m.group(1)), int(m.group(1))]),
    (r"^\(([^)]+)\)$", lambda m: [len(m.group(1).split(","))] * 2),
    (
        r"^\[([^\]]+)\]$",
        lambda m: (
            # Numeric values → per-dimension format
            [[int(v.strip()), int(v.strip())]
             for v in m.group(1).split(",")
             if v.strip().isdigit()]
            if any(v.strip().isdigit() for v in m.group(1).split(","))
            # Symbolic values (e.g. [K1, N1]) → count slots as rank
            else [len([s for s in m.group(1).split(",") if s.strip()])] * 2
        ),
    ),
    (r"^(与输入相同|同输入|same as input)$", []),
]


def _try_deterministic_parse(shape: str) -> list | None:
    """Try deterministic parsing of shape string."""
    shape_stripped = shape.strip()
    for pattern, result in _DIMENSION_PATTERNS:
        m = re.match(pattern, shape_stripped, re.IGNORECASE)
        if m:
            if callable(result):
                return result(m)
            return result
    return None


# ---------------------------------------------------------------------------
# Phase 1.5: HTML-list shape parsing (deterministic, zero LLM)
# ---------------------------------------------------------------------------

# Matches every [...] bracket group inside a shape string, e.g. in
# "<ul><li>per-channel...[E, N1]/[N1]</li><li>per-group...[E, G, N1]/[G, N1]</li></ul>"
# it extracts: "E, N1", "N1", "E, G, N1", "G, N1".
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def _try_html_list_parse(shape: str) -> list | None:
    """Handle shapes containing HTML lists with multiple dimension variants.

    These appear in quantization params where the shape depends on the mode
    (per-channel / per-group / per-tensor) and expert presence, producing
    text like::

        <ul><li>per-channel...[E, N1]/[N1]</li><li>per-group...[E, G, N1]/[G, N1]</li></ul>

    Strategy: extract **every** ``[...]`` bracket group, count the
    comma-separated slots in each (that is the rank), and return
    ``[min_rank, max_rank]`` across all variants.

    Returns ``None`` if the shape has no HTML tags or no bracket groups,
    so the caller can fall through to the LLM agent.
    """
    if "<" not in shape:
        return None
    brackets = _BRACKET_RE.findall(shape)
    if not brackets:
        return None
    counts: list[int] = []
    for bracket in brackets:
        slots = [s for s in bracket.split(",") if s.strip()]
        if slots:
            counts.append(len(slots))
    if not counts:
        return None
    return [min(counts), max(counts)]


# ---------------------------------------------------------------------------
# Phase 3: Validation (reused from dimensions_build)
# ---------------------------------------------------------------------------


def _is_rank_format(dims: list) -> bool:
    """Check if dimensions is rank format [count, count]."""
    return (
        isinstance(dims, list)
        and len(dims) == 2
        and all(isinstance(d, int) for d in dims)
    )


def _validate_dimensions_structure(dims: list) -> tuple[bool, str]:
    """Validate structure of dimensions array."""
    if not isinstance(dims, list):
        return False, "dimensions must be a list"

    if not dims:
        return True, ""

    if _is_rank_format(dims):
        min_rank, max_rank = dims[0], dims[1]
        if min_rank < 0:
            return False, f"rank min must be >= 0, got {min_rank}"
        if min_rank > max_rank:
            return False, f"rank [min, max] requires min <= max, got {dims}"
        if max_rank > 10:
            return False, f"Too many dimensions: {max_rank}"
        return True, ""

    for i, dim in enumerate(dims):
        if not isinstance(dim, list) or len(dim) != 2:
            return False, f"dim[{i}] must be [min, max], got {dim}"
        min_val, max_val = dim
        if min_val is not None and max_val is not None:
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                return False, f"dim[{i}] values must be int/float or null"
            if min_val > max_val:
                return False, f"dim[{i}]: min ({min_val}) > max ({max_val})"

    if len(dims) > 10:
        return False, f"Too many dimensions: {len(dims)}"

    return True, ""


# ---------------------------------------------------------------------------
# DeepAgent invocation for batch shape parsing
# ---------------------------------------------------------------------------


async def _parse_shapes_via_agent(shapes: list[str]) -> list[list]:
    """Use DeepAgent to parse a batch of shape strings into dimensions."""
    if not shapes:
        return []

    shape_list = "\n".join(f"{i+1}. {s}" for i, s in enumerate(shapes))
    user_msg = (
        f"Convert each shape below to dimensions. "
        f"Return a JSON array with {len(shapes)} elements, "
        f"one per shape, in order.\n\n{shape_list}"
    )

    try:
        agent = _get_dimensions_agent()
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_msg}],
        })

        msgs = result.get("messages", [])
        ai_text = ""
        for m in reversed(msgs):
            if hasattr(m, "content") and m.content:
                ai_text = m.content
                break

        parsed = _parse_dimensions_response(ai_text)

        if len(parsed) == len(shapes):
            return parsed

        logger.warning(
            "DimensionsAgent: alignment mismatch: expected %d, got %d. "
            "Falling back to per-shape parsing.",
            len(shapes), len(parsed),
        )
        return await _parse_shapes_individually(shapes)

    except Exception as e:
        logger.exception("DimensionsAgent: DeepAgent invocation failed: %s", e)
        return [[] for _ in shapes]


async def _parse_shapes_individually(shapes: list[str]) -> list[list]:
    """Parse shapes one by one via the agent (fallback for alignment mismatch)."""
    results: list[list] = []
    for shape in shapes:
        try:
            agent = _get_dimensions_agent()
            result = await agent.ainvoke({
                "messages": [{
                    "role": "user",
                    "content": (
                        "Convert this shape to dimensions. "
                        "Return ONLY a JSON array with one element.\n\n"
                        f"{shape}"
                    ),
                }],
            })
            msgs = result.get("messages", [])
            ai_text = ""
            for m in reversed(msgs):
                if hasattr(m, "content") and m.content:
                    ai_text = m.content
                    break
            parsed = _parse_dimensions_response(ai_text)
            if parsed and len(parsed) >= 1:
                results.append(parsed[0])
            else:
                results.append([])
        except Exception:
            logger.warning(
                "DimensionsAgent: per-shape parse failed for '%s'", shape,
            )
            results.append([])
    return results


def _parse_dimensions_response(text: str) -> list[list]:
    """Parse LLM/Agent response into a list of dimensions arrays."""
    data = parse_json_response(text, list)
    if not isinstance(data, list):
        logger.warning("DimensionsAgent: failed to parse response: %s", text[:200])
        return []
    # Normalize: non-list items become [] (rank specs stay as flat ints)
    result: list = []
    for item in data:
        if isinstance(item, list):
            result.append(item)
        elif isinstance(item, (int, float)):
            result.append(item)
        else:
            result.append([])
    return result


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


async def dimensions_agent_node(state: BuildParamConstraintState) -> dict[str, Any]:
    """Parse shape values into dimensions using deterministic regex + DeepAgent.

    Flow:
    1. Collect all unique shape values from params
    2. Phase 1:   Deterministic regex parsing (zero LLM, ~70-80%)
    3. Phase 1.5: HTML-list shape parsing (zero LLM, <ul><li> variants)
    4. Phase 2:   DeepAgent batch parsing for remaining shapes
    5. Phase 3:   Structure validation

    Returns:
        dimensions_map: {"fn::pn::shape_text": dimensions_array}
    """
    params = state.get("params", [])
    if not params:
        return {"dimensions_map": {}}

    # Collect all unique shape entries
    shape_entries: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for param in params:
        fn = param.get("function_name", "")
        pn = param.get("param_name", "")
        shape_json = _parse_json_field(param.get("shape", ""))
        for shape_text in shape_json.values():
            shape_text = shape_text.strip() if shape_text else ""
            if shape_text and (fn, pn, shape_text) not in seen:
                seen.add((fn, pn, shape_text))
                shape_entries.append({
                    "function_name": fn,
                    "param_name": pn,
                    "shape": shape_text,
                })

    if not shape_entries:
        return {"dimensions_map": {}}

    # Phase 1 + 1.5: Deterministic preprocessing
    result: dict[str, list] = {}
    llm_needed: list[dict] = []
    deterministic_count = 0

    for entry in shape_entries:
        # Phase 1: simple regex patterns
        deterministic = _try_deterministic_parse(entry["shape"])
        # Phase 1.5: HTML-list with multiple bracket variants
        if deterministic is None:
            deterministic = _try_html_list_parse(entry["shape"])
        if deterministic is not None:
            key = f"{entry['function_name']}::{entry['param_name']}::{entry['shape']}"
            is_valid, _ = _validate_dimensions_structure(deterministic)
            result[key] = deterministic if is_valid else []
            deterministic_count += 1
        else:
            llm_needed.append(entry)

    if deterministic_count > 0:
        logger.info(
            "DimensionsAgent: deterministic preprocessing handled %d/%d shapes",
            deterministic_count, len(shape_entries),
        )

    if not llm_needed:
        return {"dimensions_map": result}

    # Phase 2: DeepAgent batch parsing
    shapes_to_parse = [e["shape"] for e in llm_needed]
    parsed_list = await _parse_shapes_via_agent(shapes_to_parse)

    # Phase 3: Structure validation
    for i, entry in enumerate(llm_needed):
        key = f"{entry['function_name']}::{entry['param_name']}::{entry['shape']}"
        dims = parsed_list[i] if i < len(parsed_list) else []
        is_valid, validation_error = _validate_dimensions_structure(dims)
        if is_valid:
            result[key] = dims
        else:
            logger.warning(
                "DimensionsAgent: dimensions structure invalid for %s.%s: %s",
                entry["function_name"], entry["param_name"], validation_error,
            )
            result[key] = []

    logger.info(
        "DimensionsAgent: parsed %d dimensions (%d deterministic, %d agent)",
        len(result), deterministic_count, len(llm_needed),
    )

    # NODE_PROGRESS: dimensions_done — frontend ExtractorAgent panel
    from agent.runtime.context import get_context
    from agent.runtime.events import EventType, Span, SpanType
    ctx = get_context()
    if ctx and ctx.manager:
        span = Span(
            span_id="progress",
            parent_span_id=ctx.current_span_id if ctx else None,
            span_type=SpanType.NODE,
            name="build_param_constraint",
        )
        ctx.manager.emit(EventType.NODE_PROGRESS, ctx.run_id, span, {
            "agent_id": "constraint",
            "node_id": "build_param_constraint",
            "message": f"维度解析完成: {len(result)} 个 shape 已转 dimensions",
            "phase": "dimensions_done",
            "dimensions_count": len(result),
            "deterministic_count": deterministic_count,
            "agent_count": len(llm_needed),
        })

    return {"dimensions_map": result}
