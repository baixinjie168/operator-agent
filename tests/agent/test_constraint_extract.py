"""Tests for constraint_extract.py: dedup, HTML cleaning, dtype context, Pass 7.

Covers:
- _normalize_expr: commutative equivalence for A==B / len()==len() / shape==shape
- _expr_exists: dedup with normalization
- _strip_html: HTML tag removal from section content
- _pass6b_conditional_shapes_from_params: conditional shape pattern detection
- _pass7_dtype_constraints: dtype constraint generation from dtype_combos
- _extract_src_context: document-original context extraction (Item 6)
- _try_deterministic_range(return_match=True): backward-compatible match return (Item 6)
- _pass7b_conditional_dtype: conditional dtype → type_dependency (Item 4b)
"""

from __future__ import annotations

import json
import re

from agent.nodes.constraint_extract import (
    _collect_seen_exprs,
    _deduplicate_relations,
    _detect_condition_clause,
    _extract_src_context,
    _expr_exists,
    _normalize_expr,
    _pass2_single_param,
    _pass3b_dim_equalities,
    _pass6b_conditional_shapes_from_params,
    _pass6c_conditional_ordering,
    _pass7b_conditional_dtype,
    _strip_html,
    _supersede_bare_with_guarded,
    _try_deterministic_range,
)

# ── _normalize_expr ──────────────────────────────────────────────────────────

class TestNormalizeExpr:
    """Test that commutative expressions normalize to the same form."""

    def test_dtype_equality_commutive(self):
        """A.dtype == B.dtype and B.dtype == A.dtype should be equal."""
        a = _normalize_expr("x.dtype == y.dtype")
        b = _normalize_expr("y.dtype == x.dtype")
        assert a == b
        assert a == "x.dtype == y.dtype"

    def test_len_equality_commutive(self):
        a = _normalize_expr("len(x) == len(y)")
        b = _normalize_expr("len(y) == len(x)")
        assert a == b
        assert a == "len(x) == len(y)"

    def test_shape_equality_commutive(self):
        a = _normalize_expr("x.shape == y.shape")
        b = _normalize_expr("y.shape == x.shape")
        assert a == b
        assert a == "x.shape == y.shape"

    def test_non_commutative_unchanged(self):
        """Non-equality expressions should pass through unchanged."""
        assert _normalize_expr("x % 2 == 0") == "x % 2 == 0"
        assert _normalize_expr("x.range_value in [1, 2]") == "x.range_value in [1, 2]"

    def test_empty_expr(self):
        assert _normalize_expr("") == ""

    def test_three_part_equality_not_normalized(self):
        """Complex expressions with != or other ops are not normalized."""
        expr = "x.shape[0] == y.shape[0]"
        assert _normalize_expr(expr) == expr  # shape[i] not handled, passes through


# ── _expr_exists ─────────────────────────────────────────────────────────────

class TestExprExists:
    """Test dedup logic with normalization."""

    def test_detects_communicative_duplicate(self):
        """Should detect B==A as duplicate of A==B."""
        existing = [
            {"relation_object": json.dumps({"expr": "y.dtype == x.dtype"})},
        ]
        assert _expr_exists(existing, "x.dtype == y.dtype") is True

    def test_detects_len_communicative(self):
        existing = [
            {"relation_object": json.dumps({"expr": "len(y) == len(x)"})},
        ]
        assert _expr_exists(existing, "len(x) == len(y)") is True

    def test_no_false_positive(self):
        existing = [
            {"relation_object": json.dumps({"expr": "x.dtype == y.dtype"})},
        ]
        assert _expr_exists(existing, "x.dtype == z.dtype") is False

    def test_handles_string_relation_object(self):
        existing = [
            {"relation_object": {"expr": "y.shape == x.shape"}},
        ]
        assert _expr_exists(existing, "x.shape == y.shape") is True

    def test_handles_invalid_json(self):
        existing = [
            {"relation_object": "not json"},
        ]
        assert _expr_exists(existing, "x.dtype == y.dtype") is False

    def test_empty_existing(self):
        assert _expr_exists([], "x.dtype == y.dtype") is False


# ── _normalize_expr: new patterns (Item 7) ──────────────────────────────────

class TestNormalizeExprNewPatterns:
    """T7: Phase 3 Item 7 — new divisibility / multiple patterns."""

    def test_divisibility_pattern(self):
        assert _normalize_expr("oriHeight % 16 == 0") == "oriHeight % 16 == 0"

    def test_multiple_equality_pattern(self):
        assert _normalize_expr("N1 == 2 * K2") == "N1 == 2 * K2"

    def test_divisibility_with_spaces(self):
        assert _normalize_expr("x % 4 == 0") == _normalize_expr("x   %   4   ==   0")


# ── _deduplicate_relations (Item 7 — R1 platform-aware) ────────────────────

class TestDeduplicateRelations:
    """T7-*: Phase 3 Item 7 cross-source semantic dedup."""

    def _rel(self, expr, etype="cross_param_constraint", platform="", src="原文"):
        return {
            "platform": platform,
            "params": ["x", "y"],
            "relation_object": json.dumps({
                "expr": expr, "expr_type": etype, "src_text": src,
            }),
        }

    def test_t7_1_commutative_dtype_dedup(self):
        """T7-1: x.dtype==y.dtype and y.dtype==x.dtype dedup to 1."""
        new = [
            self._rel("x.dtype == y.dtype", etype="type_equality"),
            self._rel("y.dtype == x.dtype", etype="self_string_length", src="正则"),
        ]
        _, deduped = _deduplicate_relations([], new)
        assert len(deduped) == 1
        kept = json.loads(deduped[0]["relation_object"])
        assert kept["expr_type"] == "type_equality"

    def test_t7_3_guard_not_deleted(self):
        """T7-3: guarded vs unguarded have different keys, both kept."""
        new = [
            self._rel("x.shape[0] == N", etype="shape_equality"),
            self._rel("(x.shape[0] == N) if x is not None else True",
                      etype="shape_equality"),
        ]
        _, deduped = _deduplicate_relations([], new)
        assert len(deduped) == 2

    def test_t7_4_triple_repeat_dedup(self):
        """T7-4: 3 identical exprs -> 1."""
        new = [self._rel("self.shape == gradInput.shape") for _ in range(3)]
        _, deduped = _deduplicate_relations([], new)
        assert len(deduped) == 1

    def test_t7_7_existing_not_touched(self):
        """T7-7: existing relations are never removed."""
        existing = [self._rel("x.dtype == y.dtype", etype="type_equality")]
        new = [self._rel("y.dtype == x.dtype", etype="self_string_length")]
        e, n = _deduplicate_relations(existing, new)
        assert len(e) == 1  # existing unchanged
        assert len(n) == 0  # new duplicate removed

    def test_t7_platform_aware_no_cross_merge(self):
        """R1: same expr on different platforms must NOT merge."""
        new = [
            self._rel("x.shape == y.shape", platform="A2"),
            self._rel("x.shape == y.shape", platform="A3"),
        ]
        _, deduped = _deduplicate_relations([], new)
        assert len(deduped) == 2

    def test_t7_platform_common_same_platform_merges(self):
        """R1: same expr same (empty) platform -> merge."""
        new = [
            self._rel("x.shape == y.shape", platform=""),
            self._rel("x.shape == y.shape", platform=""),
        ]
        _, deduped = _deduplicate_relations([], new)
        assert len(deduped) == 1


# ── _strip_html ──────────────────────────────────────────────────────────────

class TestStripHtml:
    """Test HTML tag removal."""

    def test_removes_table_tags(self):
        assert _strip_html("<td>hello</td>") == "hello"

    def test_removes_closing_tags(self):
        assert _strip_html("</table>text<br/>") == "text"

    def test_preserves_inner_text(self):
        assert _strip_html("<p>line1</p><p>line2</p>") == "line1line2"

    def test_no_html_unchanged(self):
        assert _strip_html("no html here") == "no html here"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_none_passthrough(self):
        assert _strip_html(None) is None

    def test_nested_tags(self):
        assert _strip_html("<div><span>nested</span></div>") == "nested"

    def test_strips_with_attributes(self):
        assert _strip_html('<a href="link">text</a>') == "text"


# ── _pass6b_conditional_shapes_from_params ───────────────────────────────────

class TestPass6bConditionalShapes:
    """Test conditional shape detection from parameter descriptions."""

    def test_detects_multi_shape_candidate(self):
        """Should detect [E,K1,N1]/[K1,N1] pattern."""
        params = [
            {"param_name": "weight1", "function_name": "fn",
             "param_desc": "shape: [E,K1,N1]/[K1,N1]", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"weight1"})
        assert len(results) == 1
        assert results[0]["relation_object"]["expr_type"] == "shape_value_dependency"
        assert results[0]["relation_object"]["expr"] == ""  # empty for Agent

    def test_detects_expert_conditional(self):
        """Should detect 有专家/无专家 pattern."""
        params = [
            {"param_name": "weight1", "function_name": "fn",
             "param_desc": "有专家时3维，无专家时2维", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"weight1"})
        assert len(results) == 1

    def test_detects_quantization_conditional(self):
        """Should detect per-channel/per-tensor pattern."""
        params = [
            {"param_name": "scale", "function_name": "fn",
             "param_desc": "per-channel时为[E,N1]，per-tensor时为[N1]", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"scale"})
        assert len(results) == 1

    def test_no_pattern_no_result(self):
        """Should return empty list for params without conditional shapes."""
        params = [
            {"param_name": "x", "function_name": "fn",
             "param_desc": "普通参数描述", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"x"})
        assert len(results) == 0

    def test_dedup_against_existing(self):
        """Should skip params that already have shape_value_dependency."""
        params = [
            {"param_name": "weight1", "function_name": "fn",
             "param_desc": "[E,K1,N1]/[K1,N1]", "llm_description": ""},
        ]
        existing = [
            {"params": ["weight1"],
             "relation_object": {"expr_type": "shape_value_dependency", "expr": "x"}},
        ]
        results = _pass6b_conditional_shapes_from_params(params, existing, {"weight1"})
        assert len(results) == 0

    def test_strips_html_from_text(self):
        """Should strip HTML from param descriptions before pattern matching."""
        params = [
            {"param_name": "weight1", "function_name": "fn",
             "param_desc": "<td>[E,K1,N1]/[K1,N1]</td>", "llm_description": ""},
        ]
        results = _pass6b_conditional_shapes_from_params(params, [], {"weight1"})
        assert len(results) == 1
        # src_text should not contain HTML tags
        src = results[0]["source_citation"]
        assert "<td>" not in src


# ── _collect_seen_exprs (Item 3 helper) ──────────────────────────────────────

class TestCollectSeenExprs:
    """T3-7: extract expr set from existing relations (dict + JSON string)."""

    def test_dict_relation_object(self):
        existing = [
            {"relation_object": {"expr": "K1 == N2"}},
        ]
        seen = _collect_seen_exprs(existing)
        assert "K1 == N2" in seen

    def test_json_string_relation_object(self):
        existing = [
            {"relation_object": json.dumps({"expr": "N1 == 2 * K2"})},
        ]
        seen = _collect_seen_exprs(existing)
        assert "N1 == 2 * K2" in seen

    def test_invalid_json_skipped(self):
        existing = [
            {"relation_object": "not json"},
            {"relation_object": 123},
        ]
        seen = _collect_seen_exprs(existing)
        # Invalid JSON is skipped via `continue`; non-dict values (int) fail
        # the isinstance(obj, dict) check. Neither adds to the set.
        assert seen == set()

    def test_empty_existing(self):
        assert _collect_seen_exprs([]) == set()


# ── _detect_condition_clause ─────────────────────────────────────────────────

class TestDetectConditionClause:
    def test_activation_clause(self):
        text = "激活层为geglu/swiglu/reglu时，且N1=2*K2"
        cond = _detect_condition_clause(text, text.index("N1="))
        assert "activation.range_value in" in cond
        assert "'geglu'" in cond
        assert "'swiglu'" in cond

    def test_no_clause_returns_empty(self):
        text = "需满足K1=N2"
        cond = _detect_condition_clause(text, text.index("K1="))
        assert cond == ""

    def test_non_activation_clause_not_encoded(self):
        """当X时 without 激活 → detected but not encoded (conservative)."""
        text = "当mode为high时，K1=N2"
        cond = _detect_condition_clause(text, text.index("K1="))
        # Non-activation conditions return "" so the equality is emitted
        # unconditionally (never dropped).
        assert cond == ""

    def test_activation_clause_long_sentence(self):
        """FFNV3 bug: condition at sentence head, equality at tail (>80 chars).

        The old 80-char window missed the condition; the sentence-level window
        must reach back to the sentence head.
        """
        text = (
            "激活层为geglu/swiglu/reglu时，仅支持无专家分组时的FLOAT16高性能场景"
            "（FLOAT16场景指类型为aclTensor的必选参数数据类型都为FLOAT16的场景），"
            "且N1=2*K2。"
        )
        cond = _detect_condition_clause(text, text.index("N1="))
        assert cond == "activation.range_value in ['geglu', 'swiglu', 'reglu']"

    def test_activation_clause_quoted_values(self):
        """Guard must use quoted string literals (bare names would NameError)."""
        text = "激活层为gelu/fastgelu时，且N1=K2"
        cond = _detect_condition_clause(text, text.index("N1="))
        assert "'gelu'" in cond and "'fastgelu'" in cond
        # No bare (unquoted) gelu token left in the guard
        assert cond.replace("'gelu'", "").replace("'fastgelu'", "") == (
            "activation.range_value in [, ]"
        )

    def test_condition_not_swallowed_across_sentence(self):
        """Previous sentence's 激活 condition must not leak into next equality."""
        text = (
            "激活层为geglu时，N1=2*K2。"   # sentence 1: guarded
            "需满足N1=K2。"                 # sentence 2: no condition
        )
        cond_s2 = _detect_condition_clause(text, text.index("N1=K2"))
        assert cond_s2 == ""


# ── _pass3b_dim_equalities (Item 3) ──────────────────────────────────────────

def _implicit_param(name):
    return {
        "var_name": name,
        "is_constant": False,
        "is_external_constant": False,
        "is_quantization_type": False,
    }


class TestPass3bDimEqualities:
    """T3-1..T3-6: dimension-equality extraction."""

    def test_t3_1_simple_equality(self):
        text = "所有场景下需满足K1=N2"
        ips = [_implicit_param("K1"), _implicit_param("N2")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert len(results) == 1
        assert results[0]["relation_object"]["expr"] == "K1 == N2"
        assert results[0]["relation_type"] == "cross_param_constraint"
        assert results[0]["relation_object"]["expr_type"] == "cross_variable_equality"

    def test_t3_2_multiple_equality(self):
        text = "且N1=2*K2"
        ips = [_implicit_param("N1"), _implicit_param("K2")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert len(results) == 1
        assert results[0]["relation_object"]["expr"] == "N1 == 2 * K2"

    def test_t3_3_conditional_equality(self):
        text = "激活层为geglu/swiglu/reglu时，且N1=2*K2"
        ips = [_implicit_param("N1"), _implicit_param("K2")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert len(results) == 1
        expr = results[0]["relation_object"]["expr"]
        assert "N1 == 2 * K2" in expr
        assert "if" in expr
        assert "activation.range_value in" in expr
        assert "else True" in expr

    def test_t3_4_unknown_variable_filtered(self):
        """K1=foo where foo is not an implicit var → no constraint."""
        text = "K1=foo"
        ips = [_implicit_param("K1")]  # foo not in var_names
        results = _pass3b_dim_equalities(text, ips, [])
        assert results == []

    def test_t3_5_numeric_rhs_filtered(self):
        """K1=65536 has numeric RHS → not a dim equality (Pass 3 handles it)."""
        text = "K1=65536"
        ips = [_implicit_param("K1")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert results == []

    def test_t3_6_dedup_duplicate_equality(self):
        """K1=N2 appearing twice → only one constraint."""
        text = "需满足K1=N2，另外K1=N2再次出现"
        ips = [_implicit_param("K1"), _implicit_param("N2")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert len(results) == 1

    def test_t3_7_dedup_against_existing(self):
        """Existing constraint with same expr → new one skipped."""
        text = "需满足K1=N2"
        ips = [_implicit_param("K1"), _implicit_param("N2")]
        existing = [
            {"relation_object": {"expr": "K1 == N2"}},
        ]
        results = _pass3b_dim_equalities(text, ips, existing)
        assert results == []

    def test_t3_8_empty_sections_returns_empty(self):
        """Pass 3 failure blanks sections_text → Pass 3b returns []."""
        ips = [_implicit_param("K1"), _implicit_param("N2")]
        assert _pass3b_dim_equalities("", ips, []) == []
        assert _pass3b_dim_equalities("   ", ips, []) == []

    def test_no_implicit_params_returns_empty(self):
        assert _pass3b_dim_equalities("K1=N2", [], []) == []

    def test_nmul_matched_before_eq(self):
        """N1=2*K2 must not be split into N1=2 by the equality regex."""
        text = "N1=2*K2"
        ips = [_implicit_param("N1"), _implicit_param("K2")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert len(results) == 1
        assert results[0]["relation_object"]["expr"] == "N1 == 2 * K2"

    def test_chinese_equality(self):
        text = "K1与N2相等"
        ips = [_implicit_param("K1"), _implicit_param("N2")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert len(results) == 1
        assert results[0]["relation_object"]["expr"] == "K1 == N2"

    def test_chinese_multiple(self):
        text = "N1是K2的2倍"
        ips = [_implicit_param("N1"), _implicit_param("K2")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert len(results) == 1
        assert results[0]["relation_object"]["expr"] == "N1 == 2 * K2"

    def test_self_equality_filtered(self):
        """X=X must not produce a tautological constraint."""
        text = "K1=K1"
        ips = [_implicit_param("K1")]
        results = _pass3b_dim_equalities(text, ips, [])
        assert results == []


# ── Item 6: _extract_src_context ────────────────────────────────────────────

class TestExtractSrcContext:
    """Item 6: document-original context extraction (T6-1, T6-2)."""

    def test_t6_1_returns_context_around_match(self):
        """T6-1: returns ±50 chars of context containing the matched text."""
        text = "前面一些文字 " + "取值范围为0~100" + " 后面一些文字" * 5
        m = re.search(r"0~100", text)
        result = _extract_src_context(text, m)
        assert "0~100" in result
        assert len(result) > 5  # not empty

    def test_t6_2_match_at_edge(self):
        """T6-2: match near the edge — truncated but not empty."""
        text = "取值范围为0~100"
        m = re.search(r"0~100", text)
        result = _extract_src_context(text, m)
        assert "0~100" in result

    def test_strips_html(self):
        """HTML tags in context are removed."""
        text = "prefix <td>取值范围为0~100</td> suffix"
        m = re.search(r"0~100", text)
        result = _extract_src_context(text, m)
        assert "<td>" not in result
        assert "0~100" in result

    def test_empty_text_returns_empty(self):
        m = re.search(r"0", "0")
        assert _extract_src_context("", m) == ""

    def test_none_match_returns_empty(self):
        assert _extract_src_context("some text", None) == ""

    def test_custom_radius(self):
        text = "aaaaaaaaaa0~100bbbbbbbbbb"
        m = re.search(r"0~100", text)
        result = _extract_src_context(text, m, radius=2)
        assert "0~100" in result
        # radius=2 → only 2 chars on each side
        assert len(result) <= len("0~100") + 4


# ── Item 6: _try_deterministic_range(return_match=True) ─────────────────────

class TestTryDeterministicRangeReturnMatch:
    """Item 6: backward-compatible return_match parameter (T6-7)."""

    def test_t6_7_backward_compat_default(self):
        """Default return_match=False returns a 2-tuple (no match)."""
        result = _try_deterministic_range("取值范围为0~100")
        assert result is not None
        assert len(result) == 2  # (value_list, ar_type)

    def test_return_match_true_returns_3_tuple(self):
        """return_match=True returns a 3-tuple including the re.Match."""
        result = _try_deterministic_range("取值范围为0~100", return_match=True)
        assert result is not None
        assert len(result) == 3  # (value_list, ar_type, match)
        value_list, ar_type, match = result
        assert ar_type == "range"
        assert isinstance(match, re.Match)

    def test_no_match_returns_none(self):
        assert _try_deterministic_range("no range here", return_match=True) is None

    def test_match_context_matches_text(self):
        """The returned match's context can be extracted via _extract_src_context."""
        text = "前文 取值范围为0~100 后文"
        result = _try_deterministic_range(text, return_match=True)
        assert result is not None
        _, _, match = result
        ctx = _extract_src_context(text, match)
        assert "0~100" in ctx


# ── Item 4b: _pass7b_conditional_dtype ──────────────────────────────────────

class TestPass7bConditionalDtype:
    """Item 4b: conditional dtype → type_dependency constraint (T4-5, T4-8)."""

    def test_t4_5_generates_type_dependency(self):
        """T4-5: structured dtype JSON → type_dependency constraint."""
        params = [{
            "param_name": "x",
            "function_name": "aclnnFFNV3",
            "data_type": json.dumps({"量化": "INT8", "*": "FLOAT16"},
                                   ensure_ascii=False),
        }]
        results = _pass7b_conditional_dtype(params, [])
        assert len(results) == 1
        rel = results[0]
        obj = rel["relation_object"]
        assert obj["expr_type"] == "type_dependency"
        assert "x.dtype == 'INT8'" in obj["expr"]
        assert "quantization" in obj["expr"]
        assert "x.dtype == 'FLOAT16'" in obj["expr"]

    def test_t4_8_skipped_when_type_equality_exists(self):
        """T4-8: existing type_equality for same param → skipped."""
        existing = [{
            "function_name": "aclnnFFNV3",
            "params": ["x"],
            "relation_object": {
                "expr_type": "type_equality",
                "expr": "x.dtype == 'FLOAT16'",
                "relation_params": ["x"],
                "src_text": "",
            },
        }]
        params = [{
            "param_name": "x",
            "function_name": "aclnnFFNV3",
            "data_type": json.dumps({"量化": "INT8", "*": "FLOAT16"},
                                   ensure_ascii=False),
        }]
        results = _pass7b_conditional_dtype(params, existing)
        assert results == []

    def test_non_conditional_dtype_skipped(self):
        """Plain {"*: "FLOAT16"} has no condition branch → skipped."""
        params = [{
            "param_name": "x",
            "function_name": "aclnnOp",
            "data_type": json.dumps({"*": "FLOAT16"}),
        }]
        assert _pass7b_conditional_dtype(params, []) == []

    def test_empty_data_type_skipped(self):
        params = [{"param_name": "x", "function_name": "aclnnOp", "data_type": ""}]
        assert _pass7b_conditional_dtype(params, []) == []

    def test_non_json_data_type_skipped(self):
        params = [{"param_name": "x", "function_name": "aclnnOp",
                   "data_type": "FLOAT16"}]
        assert _pass7b_conditional_dtype(params, []) == []

    def test_dtype_desc_field_compatible(self):
        """dtype_desc field name is also accepted (agent db.py fallback)."""
        params = [{
            "param_name": "x",
            "function_name": "aclnnOp",
            "dtype_desc": json.dumps({"量化": "INT8", "*": "FLOAT16"},
                                     ensure_ascii=False),
        }]
        results = _pass7b_conditional_dtype(params, [])
        assert len(results) == 1

    def test_multiple_cond_branches(self):
        """Multiple condition branches produce 'and'-joined clauses."""
        params = [{
            "param_name": "x",
            "function_name": "aclnnOp",
            "data_type": json.dumps(
                {"量化": "INT8", "训练": "FLOAT32", "*": "FLOAT16"},
                ensure_ascii=False,
            ),
        }]
        results = _pass7b_conditional_dtype(params, [])
        assert len(results) == 1
        expr = results[0]["relation_object"]["expr"]
        assert " and " in expr

    def test_empty_params_returns_empty(self):
        assert _pass7b_conditional_dtype([], []) == []


# ── _pass2_single_param: axis length bound (F3) ──────────────────────────────

class TestPass2AxisLength:
    def _param(self, shape_text, ptype="aclTensor*"):
        return {
            "param_name": "expertTokensOptional",
            "function_name": "aclnnFFNV3GetWorkspaceSize",
            "param_type": ptype,
            "param_desc": "各专家的token数。",
            "shape": json.dumps({"common": shape_text}, ensure_ascii=False),
        }

    def test_1d_with_max_length_emits_shape0_bound(self):
        results = _pass2_single_param([self._param("1维，最大长度256")], [])
        exprs = [r["relation_object"]["expr"] for r in results]
        assert any(
            "expertTokensOptional.shape[0]" in e
            and "256" in e
            and "is not None" in e
            and "0 <" in e
            for e in exprs
        ), exprs

    def test_multi_dim_skips_axis_length(self):
        """2维 + 最大长度 must not emit a shape[0] bound (axis ambiguous)."""
        results = _pass2_single_param(
            [self._param("2维，最大长度128", ptype="aclTensor*")], [],
        )
        # rename to avoid name collision in expr match
        exprs = [r["relation_object"]["expr"] for r in results]
        assert not any("shape[0]" in e and "128" in e for e in exprs), exprs

    def test_no_max_length_no_bound(self):
        results = _pass2_single_param([self._param("1维")], [])
        exprs = [r["relation_object"]["expr"] for r in results]
        assert not any("shape[0]" in e for e in exprs), exprs


# ── _pass2_single_param: shape rank range (C2 / Fix 2A) ──────────────────────

class TestPass2ShapeRankRange:
    """Fix 2A: 'shape支持N-M维' (lo>=1) → self_shape_rank_range.

    与 _P2_SHAPE_UPPER_RE 的 'shape支持0-N维' (lo=0) 互斥；rank 区间用
    len(x.shape)，不得用 shape[0]（aclnnCalculateMatmulWeightSize 缺陷 2）。
    """

    def _param(self, desc, name="tensorShape", ptype="aclIntArray*"):
        return {
            "param_name": name,
            "function_name": "aclnnCalculateMatmulWeightSize",
            "param_type": ptype,
            "llm_description": desc,
        }

    def test_rank_range_emits_len_shape(self):
        """'shape支持2-6维' → 2 <= len(tensorShape.shape) <= 6。"""
        results = _pass2_single_param(
            [self._param("输入shape支持2-6维，即（batch，n，k）")], [],
        )
        rank = [
            r for r in results
            if r["relation_object"].get("expr_type") == "self_shape_rank_range"
        ]
        assert len(rank) == 1
        assert rank[0]["relation_object"]["expr"] == "2 <= len(tensorShape.shape) <= 6"
        assert rank[0]["relation_object"]["relation_params"] == ["tensorShape"]

    def test_lo_zero_still_upper_bound_not_range(self):
        """'shape支持0-4维' (lo=0) 归上界规则，不触发 range 规则（不重复）。"""
        results = _pass2_single_param([self._param("shape支持0-4维", name="x")], [])
        assert not any(
            r["relation_object"].get("expr_type") == "self_shape_rank_range"
            for r in results
        ), [r["relation_object"] for r in results]
        assert any(
            "len(x.shape) <= 4" in r["relation_object"].get("expr", "")
            for r in results
        ), [r["relation_object"] for r in results]

    def test_no_shape_prefix_no_range(self):
        """裸'支持2-6维'（无 shape 前缀）不误命中其它参数文本。"""
        results = _pass2_single_param([self._param("支持2-6维", name="y")], [])
        assert not any(
            r["relation_object"].get("expr_type") == "self_shape_rank_range"
            for r in results
        ), [r["relation_object"] for r in results]

    def test_rank_range_suppressed_by_existing_upper_bound(self):
        """已有 self_shape_upper_bound 时，rank range 被双向互抑跳过。"""
        existing = [{
            "params": ["x"],
            "relation_object": {
                "expr_type": "self_shape_upper_bound",
                "expr": "len(x.shape) <= 6",
                "relation_params": ["x"],
                "src_text": "",
            },
        }]
        results = _pass2_single_param(
            [self._param("shape支持2-6维", name="x")], existing,
        )
        assert not any(
            r["relation_object"].get("expr_type") == "self_shape_rank_range"
            for r in results
        ), [r["relation_object"] for r in results]

    def test_upper_bound_suppressed_by_existing_rank_range(self):
        """反向：已有 rank range 时，上界规则也被互抑跳过（双向）。"""
        existing = [{
            "params": ["x"],
            "relation_object": {
                "expr_type": "self_shape_rank_range",
                "expr": "2 <= len(x.shape) <= 6",
                "relation_params": ["x"],
                "src_text": "",
            },
        }]
        results = _pass2_single_param(
            [self._param("shape维度不高于6维", name="x")], existing,
        )
        assert not any(
            r["relation_object"].get("expr_type") == "self_shape_upper_bound"
            for r in results
        ), [r["relation_object"] for r in results]


# ── _pass6c_conditional_ordering (F4) ────────────────────────────────────────

class TestPass6cOrdering:
    _SOURCE = (
        "tokensIndexFlag为true且有专家（expertTokens不为空）时，"
        "expertTokens中的数值必须满足：如果i和j都是expertTokens中有效的数组索引，"
        "且j大于i，那么expertTokens中第j个元素的数值大于或者等于"
        "expertTokens中第i个元素的数值。"
    )

    def test_detects_non_decreasing_ordering(self):
        results = _pass6c_conditional_ordering(
            self._SOURCE,
            all_param_names={"expertTokensOptional", "tokensIndexFlag", "x"},
            implicit_param_names={"E", "N1"},
            existing=[],
        )
        assert len(results) == 1
        obj = results[0]["relation_object"]
        assert obj["expr_type"] == "self_value_ordering"
        expr = obj["expr"]
        assert "all(expertTokensOptional[i] <= expertTokensOptional[i + 1]" in expr
        assert "tokensIndexFlag.range_value == True" in expr
        assert "E > 0" in expr
        assert "expertTokensOptional is not None" in expr
        assert "len(expertTokensOptional) > 0" in expr
        assert "else True" in expr
        # relation_params MUST include all referenced params (not just et_param)
        assert "expertTokensOptional" in obj["relation_params"]
        assert "tokensIndexFlag" in obj["relation_params"]
        assert "E" in obj["relation_params"]

    def test_no_tokensflag_returns_empty(self):
        text = "expertTokens中第j个元素大于或者等于第i个元素。"
        results = _pass6c_conditional_ordering(
            text, {"expertTokensOptional"}, set(), [],
        )
        assert results == []

    def test_no_ordering_pattern_returns_empty(self):
        text = "tokensIndexFlag为true时，expertTokens为索引值。"
        results = _pass6c_conditional_ordering(
            text, {"expertTokensOptional", "tokensIndexFlag"}, {"E"}, [],
        )
        assert results == []

    def test_param_name_maps_to_longest(self):
        """Source 'expertTokens' maps to param 'expertTokensOptional'."""
        text = "tokensIndexFlag为true且有专家时，第j个元素大于或者等于第i个元素。"
        results = _pass6c_conditional_ordering(
            text, {"expertTokensOptional", "tokensIndexFlag"}, {"E"}, [],
        )
        assert len(results) == 1
        assert "expertTokensOptional" in results[0]["relation_object"]["expr"]

    def test_dedup_against_existing(self):
        results = _pass6c_conditional_ordering(
            self._SOURCE,
            {"expertTokensOptional", "tokensIndexFlag"},
            {"E"},
            existing=[{  # same expr already exists
                "relation_object": json.dumps({
                    "expr_type": "self_value_ordering",
                    "expr": "(all(expertTokensOptional[i] <= expertTokensOptional[i + 1] for i in range(len(expertTokensOptional) - 1))) if (expertTokensOptional is not None and len(expertTokensOptional) > 0 and tokensIndexFlag.range_value == True and E > 0) else True",
                    "relation_params": ["expertTokensOptional", "tokensIndexFlag", "E"],
                    "src_text": "",
                }),
            }],
        )
        assert results == []


# ── _supersede_bare_with_guarded (F1) ────────────────────────────────────────

class TestSupersedeBareWithGuarded:
    def _rel(self, expr, et="cross_variable_equality"):
        return {
            "relation_object": {
                "expr_type": et,
                "expr": expr,
                "relation_params": ["N1", "K2"],
                "src_text": "",
            },
            "platform": "",
        }

    def test_bare_removed_when_guarded_exists(self):
        existing = [self._rel("N1 == 2 * K2"), self._rel("N1 == K2")]
        new_rels = [self._rel(
            "(N1 == 2 * K2) if activation.range_value in ['geglu'] else True"
        )]
        kept = _supersede_bare_with_guarded(existing, new_rels)
        kept_exprs = [r["relation_object"]["expr"] for r in kept]
        assert "N1 == 2 * K2" not in kept_exprs  # superseded by guarded
        assert "N1 == K2" in kept_exprs  # unrelated bare kept

    def test_guarded_existing_not_removed(self):
        existing = [self._rel(
            "(N1 == K2) if activation.range_value in ['gelu'] else True"
        )]
        new_rels = [self._rel(
            "(N1 == K2) if activation.range_value in ['gelu'] else True"
        )]
        kept = _supersede_bare_with_guarded(existing, new_rels)
        assert len(kept) == 1  # guarded existing is not bare, not removed

    def test_no_guarded_new_returns_existing_unchanged(self):
        existing = [self._rel("N1 == K2")]
        new_rels = [self._rel("N1 == K2")]  # bare new, not guarded
        kept = _supersede_bare_with_guarded(existing, new_rels)
        assert len(kept) == 1
