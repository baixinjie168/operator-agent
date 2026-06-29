"""Parameter alias resolution utilities.

Resolves shorthand parameter names used in constraint expressions to actual
parameter names found in inputs/outputs.

Examples:
    expertTokens -> expertTokensOptional  (rename, 1-to-1)
    weight -> weight1, weight2             (broadcast AND, 1-to-many)

The alias mapping is maintained in a Markdown knowledge file:
    knowledge/operator-constraint-checker/param-alias.md

Query priority:
    operator-specific mapping > global default mapping > no mapping (param is actual)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default path to the alias knowledge base (relative to project root).
# File: packages/agent/src/agent/utils/param_alias.py
# parents[0]=utils, [1]=agent(pkg), [2]=src, [3]=agent(dir), [4]=packages, [5]=root
_DEFAULT_ALIAS_PATH = (
    Path(__file__).resolve().parents[5]
    / "knowledge"
    / "operator-constraint-checker"
    / "param-alias.md"
)

# Global key for default mappings in the parsed alias dict
_GLOBAL_KEY = "_default"


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

# Matches a Markdown heading: "### aclnnFFNV3" or "## Global Default"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Matches a fenced YAML code block
_YAML_BLOCK_RE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL)


def _parse_yaml_simple(text: str) -> dict[str, list[str]]:
    """Parse a minimal YAML subset (key: [val1, val2]) without external deps.

    Handles lines like:
        expertTokens: [expertTokensOptional]
        weight: [weight1, weight2]
        # comment line
        blank line
    """
    result: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().strip('"').strip("'")
        val = val.strip()
        if not val:
            continue
        # Parse [val1, val2] or single val
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            items = [v.strip().strip('"').strip("'") for v in inner.split(",") if v.strip()]
        else:
            items = [val.strip('"').strip("'")]
        result[key] = items
    return result


def _try_parse_yaml(text: str) -> dict[str, list[str]]:
    """Try to parse YAML using PyYAML if available, otherwise fall back to simple parser."""
    try:
        import yaml
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            result: dict[str, list[str]] = {}
            for k, v in parsed.items():
                if isinstance(v, list):
                    result[k] = [str(x) for x in v]
                elif v is None:
                    continue
                else:
                    result[k] = [str(v)]
            return result
    except ImportError:
        pass
    except Exception as e:
        logger.debug("PyYAML parse failed, falling back to simple parser: %s", e)
    return _parse_yaml_simple(text)


def load_alias_map(md_path: str | Path | None = None) -> dict[str, dict[str, list[str]]]:
    """Parse the alias knowledge base Markdown file.

    Extracts YAML code blocks under section headings, organizing them into:
        {
            "_default": {shorthand: [actual_params], ...},
            "aclnnFFNV3": {shorthand: [actual_params], ...},
        }

    Args:
        md_path: Path to the param-alias.md file. Defaults to the standard location.

    Returns:
        Dict mapping section name (operator name or "_default") to alias mappings.
        Empty dict if file not found or no mappings.
    """
    path = Path(md_path) if md_path else _DEFAULT_ALIAS_PATH
    if not path.exists():
        logger.warning("param_alias: alias file not found: %s", path)
        return {}

    content = path.read_text(encoding="utf-8")
    result: dict[str, dict[str, list[str]]] = {}

    # Find all headings and their positions
    headings = list(_HEADING_RE.finditer(content))

    for i, match in enumerate(headings):
        heading_text = match.group(2).strip()
        # Determine the section body: from after this heading to the next heading
        body_start = match.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        body = content[body_start:body_end]

        # Find YAML blocks in this section
        yaml_blocks = _YAML_BLOCK_RE.findall(body)
        if not yaml_blocks:
            continue

        # Merge all YAML blocks in this section
        merged: dict[str, list[str]] = {}
        for yb in yaml_blocks:
            parsed = _try_parse_yaml(yb)
            merged.update(parsed)

        if not merged:
            continue

        # Determine the section key
        section_key = _resolve_section_key(heading_text)
        if section_key:
            if section_key in result:
                result[section_key].update(merged)
            else:
                result[section_key] = merged

    logger.info("param_alias: loaded %d sections from %s", len(result), path)
    return result


def _resolve_section_key(heading_text: str) -> str | None:
    """Determine if a heading corresponds to an operator section or the global default.

    Returns:
        "_default" for global default sections
        operator name for operator-specific sections
        None for non-mapping sections (e.g. description, convention)
    """
    lower = heading_text.lower().strip()
    # Global default sections
    if "全局默认" in heading_text or "global" in lower or "default" in lower:
        return _GLOBAL_KEY
    # Skip non-operator sections
    if any(kw in heading_text for kw in ("映射语义", "查询优先级", "使用说明", "映射类型")):
        return None
    # Operator names typically start with "aclnn"
    if heading_text.startswith("aclnn"):
        return heading_text
    return None


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def resolve_alias(
    operator_name: str,
    param_name: str,
    alias_map: dict[str, dict[str, list[str]]] | None = None,
) -> list[str] | None:
    """Resolve a shorthand parameter name to actual parameter names.

    Query priority: operator-specific > global default > None (no mapping).

    Args:
        operator_name: Operator name (e.g. "aclnnFFNV3").
        param_name: The parameter name as used in the constraint expression.
        alias_map: Pre-loaded alias map (from load_alias_map). If None, auto-loads.

    Returns:
        List of actual parameter names if alias found, None if no mapping exists.
        - For 1-to-1 rename: returns [actual_name]
        - For 1-to-many broadcast: returns [actual1, actual2, ...]
    """
    if alias_map is None:
        alias_map = load_alias_map()

    # Try operator-specific first
    op_map = alias_map.get(operator_name, {})
    if param_name in op_map:
        return op_map[param_name]

    # Fall back to global default
    default_map = alias_map.get(_GLOBAL_KEY, {})
    if param_name in default_map:
        return default_map[param_name]

    return None


def is_alias(
    operator_name: str,
    param_name: str,
    alias_map: dict[str, dict[str, list[str]]] | None = None,
) -> bool:
    """Check if a parameter name is a known alias (shorthand)."""
    return resolve_alias(operator_name, param_name, alias_map) is not None


# ---------------------------------------------------------------------------
# Expression expansion
# ---------------------------------------------------------------------------

# Matches identifiers that could be parameter names in Python expressions.
_PARAM_NAME_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

# Python keywords and builtins to exclude from param name detection
_PYTHON_KEYWORDS = frozenset({
    "True", "False", "None", "and", "or", "not", "if", "else", "elif",
    "for", "in", "while", "return", "is", "all", "any", "len", "range",
    "sum", "min", "max", "abs", "ceil", "floor", "int", "float", "str",
    "list", "dict", "set", "tuple", "bool", "print", "assert", "lambda",
    "import", "from", "as", "with", "try", "except", "finally", "raise",
    "class", "def", "global", "nonlocal", "pass", "break", "continue",
    "yield", "del", "type", "format", "shape", "dtype", "range_value",
})


def extract_param_refs(expr: str) -> list[str]:
    """Extract parameter name references from a constraint expression.

    Returns unique parameter names in order of first appearance.
    Excludes Python keywords, builtins, and attribute names (after dots).
    """
    # Remove string literals to avoid extracting names from strings
    cleaned = re.sub(r"""['][^']*[']""", "", expr)
    cleaned = re.sub(r'["][^"]*["]', "", cleaned)

    # Remove dotted attribute access (e.g. x.dtype -> x)
    cleaned = re.sub(r"\.([a-zA-Z_][a-zA-Z0-9_]*)", "", cleaned)

    refs: list[str] = []
    seen: set[str] = set()
    for m in _PARAM_NAME_RE.finditer(cleaned):
        name = m.group(1)
        if name in _PYTHON_KEYWORDS:
            continue
        if name not in seen:
            seen.add(name)
            refs.append(name)
    return refs


def expand_expr(
    operator_name: str,
    expr: str,
    relation_params: list[str],
    alias_map: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[str, list[str]]:
    """Expand shorthand parameter names in expr and relation_params.

    For 1-to-1 rename: directly replace the shorthand with the actual name.
    For 1-to-many broadcast: replace with AND combination of all actual params.

    Example:
        expr = "weight.shape[-1] % 2 == 0"
        alias: weight -> [weight1, weight2]
        result expr = "(weight1.shape[-1] % 2 == 0) and (weight2.shape[-1] % 2 == 0)"

    Args:
        operator_name: Operator name.
        expr: Constraint expression string.
        relation_params: List of parameter names in relation_params field.
        alias_map: Pre-loaded alias map. If None, auto-loads.

    Returns:
        Tuple of (expanded_expr, expanded_relation_params).
    """
    if alias_map is None:
        alias_map = load_alias_map()

    expanded_expr = expr
    expanded_rp: list[str] = []

    for rp in relation_params:
        actuals = resolve_alias(operator_name, rp, alias_map)
        if actuals is None:
            # No alias -- keep as is
            expanded_rp.append(rp)
        elif len(actuals) == 1:
            # 1-to-1 rename: replace in expr and relation_params
            expanded_expr = _replace_param_name(expanded_expr, rp, actuals[0])
            expanded_rp.append(actuals[0])
        else:
            # 1-to-many broadcast: add all actuals to relation_params,
            # expand expr with AND combination
            expanded_rp.extend(actuals)
            expanded_expr = _expand_broadcast_in_expr(expanded_expr, rp, actuals)

    return expanded_expr, expanded_rp


def _replace_param_name(expr: str, old_name: str, new_name: str) -> str:
    """Replace a parameter name in an expression, respecting word boundaries."""
    pattern = r"\b" + re.escape(old_name) + r"\b"
    return re.sub(pattern, new_name, expr)


def _expand_broadcast_in_expr(expr: str, shorthand: str, actuals: list[str]) -> str:
    """Expand a 1-to-many alias in an expression using AND semantics.

    Replaces occurrences of shorthand with each actual param and combines
    with AND.

    Example:
        expr = "weight.shape[-1] % 2 == 0"
        shorthand = "weight", actuals = ["weight1", "weight2"]
        -> "(weight1.shape[-1] % 2 == 0) and (weight2.shape[-1] % 2 == 0)"
    """
    pattern = r"\b" + re.escape(shorthand) + r"\b"
    if not re.search(pattern, expr):
        # Shorthand not in expr (maybe only in relation_params), return as is
        return expr

    # Generate one sub-expression per actual param
    sub_exprs = []
    for actual in actuals:
        sub = re.sub(pattern, actual, expr)
        sub_exprs.append("(" + sub + ")")

    return " and ".join(sub_exprs)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_param_refs(
    operator_name: str,
    referenced_params: list[str],
    actual_param_names: list[str],
    alias_map: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    """Validate parameter references against actual param names, with alias support.

    Args:
        operator_name: Operator name.
        referenced_params: Parameter names extracted from expr/relation_params.
        actual_param_names: Parameter names that exist in inputs/outputs.
        alias_map: Pre-loaded alias map. If None, auto-loads.

    Returns:
        Dict with:
            "valid": True if all refs are valid (exist or are known aliases)
            "aliases_used": List of {shorthand, actuals} for resolved aliases
            "truly_invalid": List of param names that are neither actual nor alias
    """
    if alias_map is None:
        alias_map = load_alias_map()

    actual_set = set(actual_param_names)
    aliases_used: list[dict[str, list[str]]] = []
    truly_invalid: list[str] = []

    for ref in referenced_params:
        if ref in actual_set:
            continue  # Direct match, valid
        actuals = resolve_alias(operator_name, ref, alias_map)
        if actuals is not None:
            aliases_used.append({"shorthand": ref, "actuals": actuals})
        else:
            truly_invalid.append(ref)

    return {
        "valid": len(truly_invalid) == 0,
        "aliases_used": aliases_used,
        "truly_invalid": truly_invalid,
    }
