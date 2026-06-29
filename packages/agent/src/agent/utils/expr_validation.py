"""Expression validation utilities (Phase 0).

Extracted from ``build_param_relations.py`` into a shared module so that
both ``build_param_relations.py`` and ``complex_relation_agent.py`` can
import them without circular dependency.

Phase 0 provides zero-LLM-cost validation:
- Phase 0a: Syntax check: ``ast.parse(expr, mode="eval")``
- Phase 0b: Reference check: all Name/Attribute nodes must be valid
- Phase 0c: Semantic check: tautology / empty-bool / excessive redundancy

Phase 3 additions (shared across nodes):
- ``_semantic_expr_key``: AST-canonical key for semantic-equivalence dedup
  (used by constraint_extract._deduplicate_relations and the completeness
  check node). Covers commutative equivalence, guard stripping, and
  attribute equality — far richer than the regex-based ``_normalize_expr``
  that lives in constraint_extract.py.
- ``_simplify_expr`` / ``_simplify_node``: post-generation simplification
  that factors out a sub-expression repeated 3+ times inside a top-level
  ``BoolOp(And)`` or nested ``IfExp`` (FFNV3 [50] fix). Used by
  complex_relation_agent and build_param_relations retry path.
"""

from __future__ import annotations

import ast
from collections import Counter

# ---------------------------------------------------------------------------
# Constants (moved from build_param_relations.py)
# ---------------------------------------------------------------------------

_ALLOWED_ATTRS = {"shape", "dtype", "format", "range_value"}
_BUILTIN_NAMES = {
    "True", "False", "None", "len", "range",
    "all", "any", "int", "float", "str", "bool", "set",
    "min", "max", "list",
}


# ---------------------------------------------------------------------------
# Phase 0a: Syntax validation
# ---------------------------------------------------------------------------


def validate_expr_syntax(expr: str) -> tuple[bool, str]:
    """Validate expr is a legal Python expression.

    Returns:
        (is_valid, error_message)
    """
    if not expr:
        return True, ""  # Empty expression is allowed
    try:
        ast.parse(expr, mode="eval")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


# ---------------------------------------------------------------------------
# Phase 0b: Reference validation
# ---------------------------------------------------------------------------


def validate_expr_refs(
    expr: str,
    params: list[str],
    external_constants: set[str] | None = None,
    implicit_param_names: set[str] | None = None,
) -> tuple[bool, str]:
    """Validate parameter names and attributes in expr.

    Checks:
    1. All Name nodes must be in params, Python builtins,
       external_constants, or implicit_param_names
    2. All Attribute nodes must be in _ALLOWED_ATTRS
    3. Comprehension variables (e.g., 'd' in 'all(d > 0 for d in x.shape)')
       are allowed
    """
    if not expr:
        return True, ""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False, "Invalid syntax"

    param_set = set(params)
    ext_set = external_constants or set()
    implicit_set = implicit_param_names or set()

    # Collect all comprehension variables
    comprehension_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            for generator in node.generators:
                if isinstance(generator.target, ast.Name):
                    comprehension_vars.add(generator.target.id)
                elif isinstance(generator.target, ast.Tuple):
                    for elt in generator.target.elts:
                        if isinstance(elt, ast.Name):
                            comprehension_vars.add(elt.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if (
                node.id not in param_set
                and node.id not in _BUILTIN_NAMES
                and node.id not in comprehension_vars
                and node.id not in ext_set
                and node.id not in implicit_set
            ):
                return False, f"Unknown parameter: '{node.id}'"
        if isinstance(node, ast.Attribute):
            if node.attr not in _ALLOWED_ATTRS:
                return False, f"Unknown attribute: '.{node.attr}'"
    return True, ""


# ---------------------------------------------------------------------------
# Phase 0c: Semantic validity validation (zero-LLM-cost)
# ---------------------------------------------------------------------------


def validate_expr_semantic(expr: str) -> tuple[bool, str]:
    """Phase 0c: Semantic validity check (zero LLM cost).

    Detects three classes of semantically-defective expressions that would
    otherwise pass syntax + reference checks but contribute no real
    constraint:

    1. Tautology — expr is the literal ``True`` (constraint is a no-op).
    2. Empty bool — top-level ``BoolOp(and/or)`` whose operands are *all*
       ``True`` (e.g. ``True and True and True``).
    3. Excessive redundancy — the *same* ``Compare`` sub-expression
       (identical AST structure incl. variable names) repeats 3+ times,
       indicating copy-paste boilerplate (e.g. FFNV3 [50]'s
       ``deqScale1Optional is None and deqScale1Optional is None ...``).

    Returning invalid here causes ``build_param_relations`` to retry the LLM
    (up to ``expr_max_retries``) and, on persistent failure, store an empty
    expr + ``_validation_error`` marker — preferable to persisting a
    tautological constraint.

    Args:
        expr: Python expression string.

    Returns:
        ``(is_valid, error_message)``. Syntax errors are delegated to
        Phase 0a (returns valid here so 0a can produce the precise message).
    """
    if not expr:
        return True, ""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        # Delegated to Phase 0a which emits line/msg details.
        return True, ""

    body = tree.body

    # 1. Literal-True tautology
    if isinstance(body, ast.Constant) and body.value is True:
        return False, "expr 恒真(字面 True)，约束形同虚设"

    # 2. Empty bool: BoolOp whose operands are all literal True
    if isinstance(body, ast.BoolOp) and all(
        isinstance(v, ast.Constant) and v.value is True
        for v in body.values
    ):
        return False, "expr 为空布尔(仅 True 经 and/or 连接)"

    # 3. Excessive redundancy: same Compare sub-expression repeated 3+ times.
    # ast.dump(annotate_fields=False) yields a structural signature; only
    # *fully identical* sub-expressions (incl. variable names) are counted,
    # so legitimate multi-branch any()/all() expressions whose sub-items
    # differ are not affected.
    counts: dict[str, int] = {}
    sample: dict[str, ast.Compare] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        sig = ast.dump(node, annotate_fields=False)
        counts[sig] = counts.get(sig, 0) + 1
        sample.setdefault(sig, node)
    for sig, n in counts.items():
        if n >= 3:
            try:
                frag = ast.unparse(sample[sig])
            except Exception:  # noqa: BLE001
                frag = "<subexpr>"
            return False, f"expr 过度冗余: 子表达式 {frag} 重复 {n} 次"

    return True, ""


# ---------------------------------------------------------------------------
# Phase 0: Comprehensive validation (syntax + references + semantic)
# ---------------------------------------------------------------------------


def validate_expr(
    expr: str,
    params: list[str],
    external_constants: set[str] | None = None,
    implicit_param_names: set[str] | None = None,
) -> tuple[bool, str]:
    """Phase 0: Comprehensive validation (syntax + references + semantic)."""
    is_valid, error = validate_expr_syntax(expr)
    if not is_valid:
        return False, error
    is_valid, error = validate_expr_refs(
        expr, params, external_constants, implicit_param_names,
    )
    if not is_valid:
        return False, error
    # Phase 0c: semantic validity (tautology / empty-bool / redundancy)
    is_valid, error = validate_expr_semantic(expr)
    if not is_valid:
        return False, error
    return True, ""


# ---------------------------------------------------------------------------
# Phase 3: AST-canonical semantic key (for cross-source dedup)
# ---------------------------------------------------------------------------
# Lives here (not in constraint_extract.py) so that both constraint_extract
# and the constraint_completeness_check node can import it without circular
# dependency. The fallback on AST-parse failure returns the raw expr string
# (exact-match dedup) rather than calling constraint_extract._normalize_expr,
# which would create an import cycle.


def _semantic_expr_key(expr: str) -> str:
    """AST-canonical key for semantic-equivalence dedup.

    Compared to the regex-based ``_normalize_expr`` in constraint_extract.py,
    this covers many more equivalence patterns:

    - **Commutativity**: ``A == B`` and ``B == A`` → same key;
      ``A and B`` and ``B and A`` → same key (operands sorted).
    - **Guard stripping**: ``(X) if guard else True`` → ``guarded(X_key)``
      so the guarded version is recognisable as *overlapping* (but not
      strictly equal) with the unguarded ``X`` — they get different keys
      and are NOT merged (guard is more precise).
    - **Attribute equality**: ``A.dtype == B.dtype`` handled at AST level,
      covering regex-unreachable variants.

    Returns ``""`` for empty expr. On SyntaxError, returns the raw expr
    string so callers fall back to exact-string comparison (no crash).
    """
    if not expr:
        return ""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr
    return _canonicalize_node(tree.body)


def _canonicalize_node(node) -> str:  # noqa: ANN001
    """Recursively canonicalise an AST node into a semantic key string."""
    if isinstance(node, ast.BoolOp):
        op = "and" if isinstance(node.op, ast.And) else "or"
        parts = sorted(_canonicalize_node(v) for v in node.values)
        return f"{op}({','.join(parts)})"

    if isinstance(node, ast.IfExp):
        body_key = _canonicalize_node(node.body)
        orelse_key = _canonicalize_node(node.orelse)
        if orelse_key == "True":
            # (X) if guard else True  ->  guarded(X)
            return f"guarded({body_key})"
        return f"ifelse({body_key},{orelse_key})"

    if isinstance(node, ast.Compare):
        left_key = _canonicalize_node(node.left)
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            right_key = _canonicalize_node(node.comparators[0])
            pair = sorted([left_key, right_key])
            return f"eq({pair[0]},{pair[1]})"
        return ast.dump(node, annotate_fields=False)

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute):
        return f"{_canonicalize_node(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        # e.g. len(x) -> call(len,x)
        func_key = _canonicalize_node(node.func)
        arg_keys = ",".join(_canonicalize_node(a) for a in node.args)
        return f"call({func_key},{arg_keys})"
    return ast.dump(node, annotate_fields=False)


# ---------------------------------------------------------------------------
# Phase 3: post-generation simplification (factor out repeated sub-expr)
# ---------------------------------------------------------------------------
# FFNV3 [50] produced a ~2KB expr where the same Compare sub-expression
# (e.g. ``deqScale1Optional is None``) was copy-pasted 3+ times inside a
# top-level ``BoolOp(And)`` / nested ``IfExp``. validate_expr_semantic
# already *rejects* such exprs; this function attempts to *repair* them by
# extracting the repeated sub-expression as a common factor, so the retry
# path (or the initial Agent call) can succeed without another LLM round.


def _has_ifexp(expr: str) -> bool:
    """Check if *expr* contains an IfExp (conditional expression).

    Shared helper used by FIX-5 (contradiction detection) and FIX-9
    (None guard validation) to skip conditional constraints.
    """
    if not expr:
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(n, ast.IfExp) for n in ast.walk(tree))


# ---------------------------------------------------------------------------
# FIX-5: Constraint contradiction detection
# ---------------------------------------------------------------------------


def _extract_eq_targets(expr: str, param: str) -> list:
    """Extract what *param* is compared to in ``param == X`` clauses.

    Returns a list of constant values that *param* is equated to.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return []
    targets: list = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
        ):
            # Match either ``param == X`` or ``param.range_value == X``
            left = node.left
            if isinstance(left, ast.Name) and left.id == param:
                for cmp in node.comparators:
                    if isinstance(cmp, ast.Constant):
                        targets.append(cmp.value)
            elif (
                isinstance(left, ast.Attribute)
                and isinstance(left.value, ast.Name)
                and left.value.id == param
            ):
                for cmp in node.comparators:
                    if isinstance(cmp, ast.Constant):
                        targets.append(cmp.value)
    return targets


def _check_pair_contradiction(
    expr_i: str, expr_j: str, shared_params: set[str],
) -> str:
    """Check if two constraints contradict on a shared param.

    R11: skip if either expr contains IfExp (conditional constraint).
    Returns a reason string if contradiction found, empty string otherwise.
    """
    # R11: skip conditional constraints
    if _has_ifexp(expr_i) or _has_ifexp(expr_j):
        return ""
    for param in shared_params:
        ti = _extract_eq_targets(expr_i, param)
        tj = _extract_eq_targets(expr_j, param)
        if ti and tj:
            for a in ti:
                for b in tj:
                    if a != b:
                        return f"{param}=={a} vs {param}=={b}"
    return ""


def detect_constraint_contradictions(
    constraints: list[dict],
) -> list[tuple[int, int, str]]:
    """FIX-5: Detect contradictory constraint pairs.

    O(n^2) where n is the number of constraints per platform (typically < 30).
    Skips conditional constraints (containing IfExp) per R11.

    Returns a list of (index_i, index_j, reason) tuples.
    """
    contradictions: list[tuple[int, int, str]] = []
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            pi = set(constraints[i].get("relation_params", []))
            pj = set(constraints[j].get("relation_params", []))
            if not pi & pj:
                continue
            reason = _check_pair_contradiction(
                constraints[i].get("expr", ""),
                constraints[j].get("expr", ""),
                pi & pj,
            )
            if reason:
                contradictions.append((i, j, reason))
    return contradictions


# ---------------------------------------------------------------------------
# FIX-9: None guard validation (standalone, R14: does NOT modify validate_expr)
# ---------------------------------------------------------------------------


def _accesses_attr(expr: str, param_name: str) -> bool:
    """Check if *expr* accesses an attribute of *param_name*.

    Returns True if expr contains ``param_name.attr`` (e.g. ``bias.shape``).
    On SyntaxError, returns True (conservative: assume it does access).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == param_name:
                return True
    return False


def _has_none_guard(expr: str, param_name: str) -> bool:
    """Check if *expr* contains a None guard for *param_name*.

    Returns True if expr contains ``param_name is None`` or
    ``param_name is not None``.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if isinstance(node.left, ast.Name) and node.left.id == param_name:
                for cmp in node.comparators:
                    if isinstance(cmp, ast.Constant) and cmp.value is None:
                        return True
    # Also check ``param_name is not None`` patterns where param_name is on
    # the right side of ``is not`` / ``is`` comparisons
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for cmp in node.comparators:
                if isinstance(cmp, ast.Name) and cmp.id == param_name:
                    if isinstance(node.left, ast.Constant) and node.left.value is None:
                        return True
    return False


def validate_none_guard(
    expr: str,
    params: list[str],
    param_optional_map: dict[str, bool] | None = None,
) -> tuple[bool, str]:
    """FIX-9: Validate that optional params have None guards.

    R6: uses the DB ``is_optional`` field (via *param_optional_map*) instead
        of name heuristics.
    R14: standalone function — does NOT modify ``validate_expr`` signature.

    Checks if any optional parameter (per *param_optional_map*) is accessed
    via ``.shape``/``.dtype``/``.format``/``.range_value`` without a
    corresponding ``param is not None`` guard.  Skips conditional
    constraints (containing IfExp) per R6/FIX-5 shared logic.

    Args:
        expr: Python expression string.
        params: List of parameter names in the constraint.
        param_optional_map: ``{param_name: is_optional}`` from DB. When
            None or empty, returns ``(True, "")`` (no-op).

    Returns:
        ``(is_valid, error_message)``.
    """
    if not param_optional_map:
        return True, ""
    optional_params = [
        p for p in params if param_optional_map.get(p, False)
    ]
    if not optional_params:
        return True, ""
    for opt in optional_params:
        if _accesses_attr(expr, opt) and not _has_none_guard(expr, opt):
            # Skip if IfExp (conditional) — shared from FIX-5
            if _has_ifexp(expr):
                continue
            return False, f"Optional param {opt} without None guard"
    return True, ""


def _simplify_expr(expr: str) -> str:
    """Post-generation simplification: factor out a sub-expression repeated
    3+ times inside a top-level ``BoolOp(And)`` or nested ``IfExp``.

    Returns the simplified expr string on success, or the original expr if
    no simplification applied / simplification produced invalid syntax.

    Safety:
    - Only triggers for exprs longer than 100 chars (short exprs left alone).
    - Only factors when the *same* Compare sub-expression (identical AST)
      repeats 3+ times — legitimate multi-branch ``any()/all()`` whose
      sub-items differ are NOT affected.
    - The simplified result is re-parsed with ``ast.parse`` to guarantee
      syntax validity before returning; otherwise the original is returned.
    """
    if not expr or len(expr) < 100:
        return expr
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr
    new_body = _simplify_node(tree.body)
    if new_body is None:
        return expr
    try:
        simplified = ast.unparse(new_body)
        ast.parse(simplified, mode="eval")
        return simplified
    except Exception:  # noqa: BLE001
        return expr


def _simplify_node(node):  # noqa: ANN001
    """Recursively simplify an AST node, returning a new node or None.

    Recursion descends into ``IfExp.body`` / ``IfExp.orelse`` so that
    FFNV3 [50]'s nested-if form is also reachable (the initial version only
    inspected the top-level ``BoolOp(And)`` and missed the nested case).
    """
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        values: list = []
        changed = False
        for v in node.values:
            sv = _simplify_node(v)
            values.append(sv if sv is not None else v)
            if sv is not None:
                changed = True
        sigs = [ast.dump(v, annotate_fields=False) for v in values]
        counts = Counter(sigs)
        common_sig = next((s for s in sigs if counts[s] >= 3), None)
        if common_sig is not None:
            common_node = next(
                v for v in values
                if ast.dump(v, annotate_fields=False) == common_sig
            )
            unique, seen = [], set()
            for v in values:
                s = ast.dump(v, annotate_fields=False)
                if s == common_sig:
                    continue
                if s not in seen:
                    unique.append(v)
                    seen.add(s)
            if unique:
                tail = (
                    ast.BoolOp(op=ast.And(), values=unique)
                    if len(unique) > 1
                    else unique[0]
                )
                return ast.BoolOp(op=ast.And(), values=[common_node, tail])
        return ast.BoolOp(op=ast.And(), values=values) if changed else None

    if isinstance(node, ast.IfExp):
        body = _simplify_node(node.body)
        orelse = _simplify_node(node.orelse)
        if body is None and orelse is None:
            return None
        return ast.IfExp(
            test=node.test,
            body=body if body is not None else node.body,
            orelse=orelse if orelse is not None else node.orelse,
        )
    return None
