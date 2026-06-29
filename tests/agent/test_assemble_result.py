"""Tests for the assemble_result node."""

from __future__ import annotations

from agent.nodes.assemble_result import (
    _build_allowed_range_lookup,
    _build_ar_lookup_from_io,
    _build_constraints_in_parameters,
    _build_function_explanation,
    _build_inputs_outputs,
    _ensure_complete_range,
    _has_meaningful_expr,
    _is_compound_expr,
    _parse_range_expr,
    _transform_return_codes,
)
from agent.utils.param_validators import (
    get_primary_function_names,
    is_single_function_mode,
)


class TestBuildFunctionExplanation:
    def test_empty_inputs(self):
        result = _build_function_explanation([], [], [], [], [])
        assert result == {}

    def test_single_function_with_params(self):
        params = [
            {"function_name": "aclnnFoo", "param_name": "x", "param_type": "aclTensor"},
            {"function_name": "aclnnFoo", "param_name": "y", "param_type": "aclTensor"},
        ]
        result = _build_function_explanation(params, [], [], [], [])
        assert "aclnnFoo" in result
        assert len(result["aclnnFoo"]["params"]) == 2
        assert result["aclnnFoo"]["relations"] == []
        assert result["aclnnFoo"]["signature"] == {}

    def test_multiple_functions_grouped(self):
        params = [
            {"function_name": "aclnnFooGetWorkspaceSize", "param_name": "x"},
            {"function_name": "aclnnFoo", "param_name": "x"},
            {"function_name": "aclnnFoo", "param_name": "y"},
        ]
        result = _build_function_explanation(params, [], [], [], [])
        assert len(result) == 2
        assert "aclnnFoo" in result
        assert "aclnnFooGetWorkspaceSize" in result
        assert len(result["aclnnFoo"]["params"]) == 2
        assert len(result["aclnnFooGetWorkspaceSize"]["params"]) == 1

    def test_signature_matched(self):
        signatures = [
            {"function_name": "aclnnFoo", "full_signature": "aclnnStatus aclnnFoo(...)", "return_type": "aclnnStatus"},
        ]
        result = _build_function_explanation([], [], signatures, [], [])
        assert "aclnnFoo" in result
        assert result["aclnnFoo"]["signature"]["full_signature"] == "aclnnStatus aclnnFoo(...)"

    def test_relations_grouped(self):
        relations = [
            {"function_name": "aclnnFoo", "relation_type": "dtype", "description": "x1 and x2 must match",
             "relation_object": {"expr_type": "type_equality", "expr": "x1.dtype == x2.dtype"}},
            {"function_name": "aclnnFoo", "relation_type": "shape", "description": "same shape",
             "relation_object": {"expr_type": "shape_equality", "expr": "x1.shape == x2.shape"}},
            {"function_name": "aclnnBar", "relation_type": "dtype", "description": "bar relation",
             "relation_object": {"expr_type": "type_equality", "expr": "a.dtype == b.dtype"}},
        ]
        result = _build_function_explanation([], relations, [], [], [])
        assert len(result["aclnnFoo"]["relations"]) == 2
        assert len(result["aclnnBar"]["relations"]) == 1

    def test_relations_empty_expr_filtered(self):
        """Relations with empty expr should be excluded from output."""
        relations = [
            {"function_name": "aclnnFoo", "relation_type": "shape_broadcast",
             "relation_object": {"expr_type": "shape_broadcast", "expr": "all(x.shape[i] == y.shape[i] for i in range(len(x.shape)))"}},
            {"function_name": "aclnnFoo", "relation_type": "presence_dependency",
             "relation_object": {"expr_type": "presence_dependency", "expr": ""}},
            {"function_name": "aclnnFoo", "relation_type": "presence_dependency",
             "relation_object": {"expr_type": "presence_dependency", "expr": "  "}},
        ]
        result = _build_function_explanation([], relations, [], [], [])
        assert len(result["aclnnFoo"]["relations"]) == 1
        assert result["aclnnFoo"]["relations"][0]["relation_object"]["expr_type"] == "shape_broadcast"

    def test_return_codes_grouped(self):
        return_codes = [
            {"function_name": "aclnnFoo", "return_value": "ACLNN_ERR_NULLPTR", "error_code": 161001},
        ]
        result = _build_function_explanation([], [], [], return_codes, [])
        assert len(result["aclnnFoo"]["return_codes"]) == 1
        assert result["aclnnFoo"]["return_codes"][0]["error_code"] == 161001

    def test_dtype_combinations_grouped(self):
        combos = [
            {"function_name": "aclnnFoo", "platform": "common", "combo": {"x1": "FLOAT16"}},
        ]
        result = _build_function_explanation([], [], [], [], combos)
        assert len(result["aclnnFoo"]["dtype_combinations"]) == 1

    def test_all_sources_combined(self):
        params = [{"function_name": "aclnnFoo", "param_name": "x"}]
        relations = [{"function_name": "aclnnFoo", "relation_type": "dtype",
                      "relation_object": {"expr_type": "type_equality", "expr": "x.dtype == y.dtype"}}]
        signatures = [{"function_name": "aclnnFoo", "full_signature": "sig"}]
        return_codes = [{"function_name": "aclnnFoo", "error_code": 1}]
        combos = [{"function_name": "aclnnFoo", "combo": {}}]
        result = _build_function_explanation(params, relations, signatures, return_codes, combos)
        fn = result["aclnnFoo"]
        assert len(fn["params"]) == 1
        assert len(fn["relations"]) == 1
        assert fn["signature"]["full_signature"] == "sig"
        assert len(fn["return_codes"]) == 1
        assert len(fn["dtype_combinations"]) == 1

    def test_fn_names_sorted(self):
        params = [
            {"function_name": "zFunc", "param_name": "x"},
            {"function_name": "aFunc", "param_name": "x"},
        ]
        result = _build_function_explanation(params, [], [], [], [])
        keys = list(result.keys())
        assert keys == ["aFunc", "zFunc"]

    def test_description_injected_at_top_level(self):
        params = [{"function_name": "aclnnFoo", "param_name": "x"}]
        result = _build_function_explanation(
            params, [], [], [], [],
            description="这是一个测试算子描述",
        )
        assert result["description"] == "这是一个测试算子描述"
        assert "aclnnFoo" in result
        assert len(result["aclnnFoo"]["params"]) == 1

    def test_empty_description_omitted(self):
        result = _build_function_explanation([], [], [], [], [])
        assert "description" not in result

    def test_description_with_no_functions(self):
        result = _build_function_explanation(
            [], [], [], [], [],
            description="仅有描述无函数",
        )
        assert result == {"description": "仅有描述无函数"}


class TestTransformReturnCodes:
    def test_return_codes_transformed(self):
        """Two rows with same (return_value, error_code) from different functions merge."""
        raw = [
            {
                "id": 1,
                "function_name": "BmmV2",
                "return_value": "ACLNN_ERR_PARAM_NULLPTR",
                "error_code": 161001,
                "descriptions": ["传入的self、batch1或batch2是空指针"],
                "source_citation": "...",
            },
            {
                "id": 2,
                "function_name": "BmmV2GetWorkspaceSize",
                "return_value": "ACLNN_ERR_PARAM_NULLPTR",
                "error_code": 161001,
                "descriptions": ["传入的self、batch1或batch2是空指针"],
                "source_citation": "...",
            },
        ]
        result = _transform_return_codes(raw)
        assert len(result) == 1
        assert result[0]["return_value"] == "ACLNN_ERR_PARAM_NULLPTR"
        assert result[0]["error_code"] == 161001
        assert len(result[0]["description"]) == 2
        # id, function_name, source_citation must be stripped
        assert "id" not in result[0]
        assert "function_name" not in result[0]
        assert "source_citation" not in result[0]

    def test_return_codes_empty(self):
        """Empty input returns empty list."""
        assert _transform_return_codes([]) == []

    def test_return_codes_no_dup(self):
        """Different error_codes are not merged."""
        raw = [
            {
                "function_name": "BmmV2",
                "return_value": "ACLNN_ERR_PARAM_NULLPTR",
                "error_code": 161001,
                "descriptions": ["空指针"],
            },
            {
                "function_name": "BmmV2",
                "return_value": "ACLNN_ERR_PARAM_INVLAID",
                "error_code": 161002,
                "descriptions": ["类型不支持"],
            },
        ]
        result = _transform_return_codes(raw)
        assert len(result) == 2
        assert result[0]["error_code"] == 161001
        assert result[1]["error_code"] == 161002
        assert result[0]["description"] == ["空指针"]
        assert result[1]["description"] == ["类型不支持"]


class TestBuildInputsOutputs:
    def test_only_workspace_size_function_included(self):
        """Only params from WorkspaceSize-ending functions are included."""
        params = [
            {
                "function_name": "aclnnFooGetWorkspaceSize",
                "param_name": "x",
                "direction": "input",
                "param_constraint": '{"Atlas A2": {"type": {"value": "aclTensor"}}}',
            },
            {
                "function_name": "aclnnFoo",
                "param_name": "y",
                "direction": "input",
                "param_constraint": '{"Atlas A2": {"type": {"value": "aclTensor"}}}',
            },
        ]
        inputs, outputs = _build_inputs_outputs(params)
        assert "x" in inputs
        assert "y" not in inputs

    def test_workspace_size_and_executor_excluded(self):
        """workspaceSize and executor params are excluded."""
        params = [
            {
                "function_name": "aclnnFooGetWorkspaceSize",
                "param_name": "workspaceSize",
                "direction": "output",
                "param_constraint": "{}",
            },
            {
                "function_name": "aclnnFooGetWorkspaceSize",
                "param_name": "executor",
                "direction": "input",
                "param_constraint": "{}",
            },
            {
                "function_name": "aclnnFooGetWorkspaceSize",
                "param_name": "x",
                "direction": "input",
                "param_constraint": '{"Atlas A2": {"type": {"value": "aclTensor"}}}',
            },
        ]
        inputs, outputs = _build_inputs_outputs(params)
        assert "workspaceSize" not in outputs
        assert "executor" not in inputs
        assert "x" in inputs

    def test_input_output_split(self):
        """Params are split correctly by direction."""
        params = [
            {
                "function_name": "aclnnFooGetWorkspaceSize",
                "param_name": "x",
                "direction": "input",
                "param_constraint": '{"Atlas A2": {"type": {"value": "aclTensor"}}}',
            },
            {
                "function_name": "aclnnFooGetWorkspaceSize",
                "param_name": "y",
                "direction": "output",
                "param_constraint": '{"Atlas A2": {"type": {"value": "aclTensor"}}}',
            },
        ]
        inputs, outputs = _build_inputs_outputs(params)
        assert "x" in inputs
        assert "y" in outputs
        assert "x" not in outputs
        assert "y" not in inputs

    def test_empty_params(self):
        """Empty params returns empty dicts."""
        inputs, outputs = _build_inputs_outputs([])
        assert inputs == {}
        assert outputs == {}

    def test_constraint_parsed_from_json_string(self):
        """param_constraint is parsed from JSON string."""
        params = [
            {
                "function_name": "aclnnFooGetWorkspaceSize",
                "param_name": "x",
                "direction": "input",
                "param_constraint": '{"Atlas A2": {"type": {"value": "aclTensor"}}}',
            },
        ]
        inputs, _ = _build_inputs_outputs(params)
        assert inputs["x"] == {"Atlas A2": {"type": {"value": "aclTensor"}}}

    def test_constraint_handles_invalid_json(self):
        """Invalid JSON in param_constraint falls back to empty dict."""
        params = [
            {
                "function_name": "aclnnFooGetWorkspaceSize",
                "param_name": "x",
                "direction": "input",
                "param_constraint": "not-valid-json",
            },
        ]
        inputs, _ = _build_inputs_outputs(params)
        assert inputs["x"] == {}


class TestQuantizationTypeImplicitParam:
    """Verify quantization_type default implicit param flows to inputs."""

    def test_quantization_type_with_modes(self):
        mapping = [{
            "var_name": "quantization_type",
            "is_quantization_type": True,
            "param_type": "char",
            "allowed_range_value": ["per-channel", "per-tensor"],
            "allowed_range_type": "enum",
            "tensor_param": None,
            "dim_index": None,
            "shape_text": None,
            "is_constant": False,
            "is_external_constant": False,
        }]
        inputs, _ = _build_inputs_outputs([], implicit_params=mapping)
        assert "quantization_type" in inputs
        plat_constraint = inputs["quantization_type"]["common"]
        assert plat_constraint["type"]["value"] == "char"
        arv = plat_constraint["allowed_range_value"]
        assert arv["type"] == "enum"
        assert arv["value"] == ["per-channel", "per-tensor"]

    def test_quantization_type_empty_modes(self):
        """No document hits → empty allowed_range_value, enum type preserved."""
        mapping = [{
            "var_name": "quantization_type",
            "is_quantization_type": True,
            "param_type": "char",
            "allowed_range_value": [],
            "allowed_range_type": "enum",
            "tensor_param": None,
            "dim_index": None,
            "is_constant": False,
            "is_external_constant": False,
        }]
        inputs, _ = _build_inputs_outputs([], implicit_params=mapping)
        arv = inputs["quantization_type"]["common"]["allowed_range_value"]
        assert arv["type"] == "enum"
        assert arv["value"] == []

    def test_quantization_type_does_not_shadow_shape_dims(self):
        """A regular shape-dim implicit param keeps int64_t + range defaults."""
        mappings = [
            {
                "var_name": "BS",
                "tensor_param": "x1",
                "dim_index": 0,
                "shape_text": "(BS, H)",
                "is_constant": False,
                "is_external_constant": False,
            },
            {
                "var_name": "quantization_type",
                "is_quantization_type": True,
                "param_type": "char",
                "allowed_range_value": ["per-group"],
                "allowed_range_type": "enum",
                "tensor_param": None,
                "dim_index": None,
                "is_constant": False,
                "is_external_constant": False,
            },
        ]
        inputs, _ = _build_inputs_outputs([], implicit_params=mappings)
        # Shape dim stays int64_t / range
        bs = inputs["BS"]["common"]
        assert bs["type"]["value"] == "int64_t"
        assert bs["allowed_range_value"]["type"] == "range"
        # Quantization type is char / enum
        qt = inputs["quantization_type"]["common"]
        assert qt["type"]["value"] == "char"
        assert qt["allowed_range_value"]["type"] == "enum"
        assert qt["allowed_range_value"]["value"] == ["per-group"]


class TestHasMeaningfulExpr:
    def test_empty_string(self):
        assert _has_meaningful_expr({"expr": ""}) is False

    def test_whitespace_only(self):
        assert _has_meaningful_expr({"expr": "   "}) is False

    def test_non_empty(self):
        assert _has_meaningful_expr({"expr": "x.shape == y.shape"}) is True

    def test_missing_expr_key(self):
        assert _has_meaningful_expr({"expr_type": "presence_dependency"}) is False

    def test_empty_dict(self):
        assert _has_meaningful_expr({}) is False

    def test_non_dict(self):
        # Non-dict objects are considered meaningful (defensive)
        assert _has_meaningful_expr("some string") is True
        assert _has_meaningful_expr(42) is True

    def test_non_string_expr(self):
        # Non-string expr (e.g. list) is considered meaningful
        assert _has_meaningful_expr({"expr": [1, 2, 3]}) is True


class TestBuildConstraintsInParameters:
    def test_empty_expr_filtered(self):
        """Relations with empty expr should not appear in constraints_in_parameters."""
        relations = [
            {"platform": "", "relation_object": {
                "expr_type": "shape_broadcast",
                "expr": "all(x.shape[i] == y.shape[i] for i in range(len(x.shape)))",
            }},
            {"platform": "", "relation_object": {
                "expr_type": "presence_dependency",
                "expr": "",
            }},
        ]
        supported = ["Atlas A2 训练系列产品/Atlas A2 推理系列产品"]
        result = _build_constraints_in_parameters(relations, supported, [])
        plat = "Atlas A2 训练系列产品/Atlas A2 推理系列产品"
        assert len(result.get(plat, [])) == 1
        assert result[plat][0]["expr_type"] == "shape_broadcast"

    def test_all_empty_expr_returns_empty(self):
        """When all relations have empty expr, result is empty dict."""
        relations = [
            {"platform": "", "relation_object": {
                "expr_type": "presence_dependency",
                "expr": "",
            }},
            {"platform": "", "relation_object": {
                "expr_type": "presence_dependency",
                "expr": "",
            }},
        ]
        supported = ["Atlas A2"]
        result = _build_constraints_in_parameters(relations, supported, [])
        assert result == {}

    def test_empty_relation_object_skipped(self):
        """Relations with empty relation_object ({}) are skipped."""
        relations = [
            {"platform": "", "relation_object": {}},
            {"platform": "", "relation_object": {
                "expr_type": "type_equality",
                "expr": "x.dtype == y.dtype",
            }},
        ]
        supported = ["Atlas A2"]
        result = _build_constraints_in_parameters(relations, supported, [])
        assert len(result.get("Atlas A2", [])) == 1

    def test_dedup_single_param_value_dependency_with_allowed_range(self):
        """Single-param value_dependency is skipped when allowed_range_value covers it."""
        import json

        params = [
            {
                "param_name": "self",
                "param_constraint": json.dumps({
                    "Atlas A2": {
                        "allowed_range_value": {"value": [[0, 1]], "src_text": ""},
                    }
                }),
            },
        ]
        relations = [
            {"platform": "", "relation_object": {
                "expr_type": "value_dependency",
                "expr": "0 <= self.range_value <= 1",
                "relation_params": ["self"],
                "src_text": "取值在0~1之间",
            }},
            {"platform": "", "relation_object": {
                "expr_type": "shape_equality",
                "expr": "target.shape == self.shape",
                "relation_params": ["target", "self"],
                "src_text": "维度与self一致",
            }},
        ]
        supported = ["Atlas A2"]
        result = _build_constraints_in_parameters(relations, supported, params)
        plat = "Atlas A2"
        # value_dependency for self should be deduped, only shape_equality remains
        assert len(result.get(plat, [])) == 1
        assert result[plat][0]["expr_type"] == "shape_equality"

    def test_dedup_keeps_value_dependency_without_allowed_range(self):
        """Single-param value_dependency is kept when allowed_range_value is empty."""
        import json

        params = [
            {
                "param_name": "reduction",
                "param_constraint": json.dumps({
                    "Atlas A2": {
                        "allowed_range_value": {"value": [], "src_text": ""},
                    }
                }),
            },
        ]
        relations = [
            {"platform": "", "relation_object": {
                "expr_type": "value_dependency",
                "expr": "reduction.range_value in (0, 1, 2)",
                "relation_params": ["reduction"],
                "src_text": "枚举值0/1/2",
            }},
        ]
        supported = ["Atlas A2"]
        result = _build_constraints_in_parameters(relations, supported, params)
        plat = "Atlas A2"
        # Should be kept because allowed_range_value is empty
        assert len(result.get(plat, [])) == 1
        assert result[plat][0]["expr_type"] == "value_dependency"

    def test_dedup_ignores_multi_params_value_dependency(self):
        """Multi-param value_dependency is never deduped even if one param has allowed_range."""
        import json

        params = [
            {
                "param_name": "self",
                "param_constraint": json.dumps({
                    "Atlas A2": {
                        "allowed_range_value": {"value": [[0, 1]], "src_text": ""},
                    }
                }),
            },
        ]
        relations = [
            {"platform": "", "relation_object": {
                "expr_type": "value_dependency",
                "expr": "target.range_value == self.range_value",
                "relation_params": ["target", "self"],
                "src_text": "target取值与self一致",
            }},
        ]
        supported = ["Atlas A2"]
        result = _build_constraints_in_parameters(relations, supported, params)
        plat = "Atlas A2"
        # Multi-param value_dependency should NOT be deduped
        assert len(result.get(plat, [])) == 1
        assert result[plat][0]["expr_type"] == "value_dependency"


class TestParseRangeExpr:
    """Unit tests for _parse_range_expr expr → (value_list, ar_type)."""

    def test_chained_inclusive(self):
        """Chained 'LO <= var.range_value <= HI' keeps both bounds."""
        assert _parse_range_expr("0 <= BS.range_value <= 2147483647", "value_dependency") == (
            [[0, 2147483647]], "range",
        )

    def test_chained_inclusive_lower_one(self):
        assert _parse_range_expr("1 <= N.range_value <= 2147483647", "value_dependency") == (
            [[1, 2147483647]], "range",
        )

    def test_chained_exclusive_adjusts_by_one(self):
        """Exclusive '<' bounds are adjusted to inclusive [lo, hi]."""
        assert _parse_range_expr("7 < x.range_value < 32", "value_dependency") == (
            [[8, 31]], "range",
        )

    def test_chained_mixed_operators(self):
        assert _parse_range_expr("1 <= x.range_value < 32", "value_dependency") == (
            [[1, 31]], "range",
        )
        assert _parse_range_expr("1 < x.range_value <= 32", "value_dependency") == (
            [[2, 32]], "range",
        )

    def test_split_ge_le_form(self):
        """Split 'var >= LO and var <= HI' still works (no regression)."""
        assert _parse_range_expr(
            "BS.range_value >= 0 and BS.range_value <= 2147483647", "value_dependency",
        ) == ([[0, 2147483647]], "range")

    def test_one_sided_upper_leaves_null(self):
        """One-sided 'var <= HI' leaves lo=None for completeness enforcement."""
        assert _parse_range_expr("x.range_value <= 100", "value_dependency") == (
            [[None, 100]], "range",
        )

    def test_one_sided_lower_leaves_null(self):
        assert _parse_range_expr("x.range_value >= 1", "value_dependency") == (
            [[1, None]], "range",
        )

    def test_enum_numeric(self):
        assert _parse_range_expr("rankSize.range_value in [2, 4, 8, 16]", "value_dependency") == (
            [2, 4, 8, 16], "enum",
        )

    def test_enum_string(self):
        assert _parse_range_expr("x.range_value in ['ND', 'NZ']", "value_dependency") == (
            ["ND", "NZ"], "enum",
        )

    def test_bool_false(self):
        assert _parse_range_expr("transposeX1.range_value == False", "value_dependency") == (
            [False], "range",
        )

    def test_bool_true(self):
        assert _parse_range_expr("x.range_value == True", "self_value_dependency") == (
            [True], "range",
        )

    def test_compound_product_not_chained_parsed(self):
        """A product expr like '1 <= H*rankSize <= 35000' must NOT be parsed
        as a simple two-sided range here (lo would be None); it is filtered
        upstream by _is_compound_expr."""
        val, ar_type = _parse_range_expr(
            "1 <= H.range_value * rankSize.range_value <= 35000", "value_dependency",
        )
        assert ar_type == "range"
        # lo is None: the chained matcher cannot bind because of the '*' gap
        assert val == [[None, 35000]]

    def test_unrecognized_returns_none(self):
        assert _parse_range_expr("x.shape == y.shape", "shape_equality") == (None, "")


class TestIsCompoundExpr:
    def test_product_is_compound(self):
        assert _is_compound_expr("1 <= H.range_value * rankSize.range_value <= 35000", "H") is True

    def test_mod_is_compound(self):
        assert _is_compound_expr("BS.range_value % rankSize.range_value == 0", "BS") is True

    def test_other_param_referenced_is_compound(self):
        assert _is_compound_expr("target.range_value == self.range_value", "target") is True

    def test_single_var_not_compound(self):
        assert _is_compound_expr("0 <= BS.range_value <= 2147483647", "BS") is False

    def test_one_sided_single_var_not_compound(self):
        assert _is_compound_expr("x.range_value <= 100", "x") is False


class TestEnsureCompleteRange:
    def test_complete_range_passes(self):
        assert _ensure_complete_range([[0, 2147483647]], "range", "", "BS") == [[0, 2147483647]]

    def test_one_sided_recovered_from_src_text(self):
        """Missing lower bound recovered from '不得小于1'."""
        src = "X不得小于1，且不得超过100。"
        assert _ensure_complete_range([[None, 100]], "range", src, "X") == [[1, 100]]

    def test_one_sided_upper_recovered_from_src_text(self):
        # Each bound's clause repeats the param name (realistic doc pattern);
        # clause isolation splits on commas so the upper clause is reached.
        src = "X不得小于1，X不得超过100。"
        assert _ensure_complete_range([[1, None]], "range", src, "X") == [[1, 100]]

    def test_one_sided_unrecoverable_is_dropped(self):
        """No recoverable bound → entry dropped → None (no [null, x])."""
        src = "X不得超过100。"
        assert _ensure_complete_range([[None, 100]], "range", src, "X") is None

    def test_no_null_emitted_invariant(self):
        """Result never contains a None bound."""
        for result in (
            _ensure_complete_range([[None, 100]], "range", "X不得小于5", "X"),
            _ensure_complete_range([[None, 100]], "range", "X不超过100", "X"),
        ):
            if result:
                for lo, hi in result:
                    assert lo is not None
                    assert hi is not None

    def test_enum_passes_through_unchanged(self):
        assert _ensure_complete_range([2, 4, 8], "enum", "", "x") == [2, 4, 8]

    def test_bool_passes_through_unchanged(self):
        assert _ensure_complete_range([False], "range", "", "x") == [False]


class TestBuildAllowedRangeLookup:
    """End-to-end tests for the allowed_range_value extraction invariants."""

    def _rel(self, expr, params, src="", expr_type="value_dependency"):
        return {"relation_object": {
            "expr_type": expr_type, "expr": expr,
            "relation_params": params, "src_text": src,
        }}

    def test_chained_bs_n_recovers_lower_bound(self):
        """BS/N chained 'LO <= var <= HI' → complete range (regression for
        the original bug where the lower bound was lost → [[None, x]])."""
        src = "BS和N的值不得超过2147483647（INT32_MAX），BS的值不得小于0，N的值不得小于1。"
        rels = [
            self._rel("0 <= BS.range_value <= 2147483647", ["BS"], src),
            self._rel("1 <= N.range_value <= 2147483647", ["N"], src),
        ]
        lookup = _build_allowed_range_lookup(rels)
        assert lookup["BS"] == ([[0, 2147483647]], "range", src)
        assert lookup["N"] == ([[1, 2147483647]], "range", src)

    def test_compound_h_ranksize_is_filtered_out(self):
        """Multi-param compound '1 <= H*rankSize <= 35000' is NOT flattened
        into H's allowed_range_value (35000 is the product's bound, not H's)."""
        rels = [self._rel(
            "1 <= H.range_value * rankSize.range_value <= 35000",
            ["H", "rankSize"], "H*rankSize范围：支持[1, 35000]。",
        )]
        lookup = _build_allowed_range_lookup(rels)
        assert "H" not in lookup
        assert "rankSize" not in lookup

    def test_one_sided_unrecoverable_yields_empty(self):
        """One-sided expr with no recoverable bound → not in lookup (AR stays [])."""
        rels = [self._rel("x.range_value <= 100", ["x"], "x不得超过100。")]
        assert _build_allowed_range_lookup(rels) == {}

    def test_one_sided_recoverable_fills_both(self):
        rels = [self._rel("x.range_value <= 100", ["x"], "x不得小于1，且不得超过100。")]
        lookup = _build_allowed_range_lookup(rels)
        assert lookup["x"] == ([[1, 100]], "range", "x不得小于1，且不得超过100。")

    def test_split_bounds_across_relations_aggregate(self):
        """Bounds split across two relations are reunited into one range."""
        rels = [
            self._rel("x.range_value >= 1", ["x"], "x不得小于1"),
            self._rel("x.range_value <= 100", ["x"], "x不得超过100"),
        ]
        lookup = _build_allowed_range_lookup(rels)
        assert lookup["x"][0] == [[1, 100]]
        assert lookup["x"][1] == "range"

    def test_enum_kept(self):
        rels = [self._rel("rankSize.range_value in [2, 4, 8, 16]", ["rankSize"], "")]
        lookup = _build_allowed_range_lookup(rels)
        assert lookup["rankSize"] == ([2, 4, 8, 16], "enum", "")

    def test_bool_kept(self):
        rels = [self._rel("transposeX1.range_value == False", ["transposeX1"], "")]
        lookup = _build_allowed_range_lookup(rels)
        assert lookup["transposeX1"] == ([False], "range", "")

    def test_no_null_bounds_anywhere(self):
        """Invariant: no lookup value ever contains a None bound."""
        rels = [
            self._rel("0 <= BS.range_value <= 2147483647", ["BS"], "BS不得小于0"),
            self._rel("1 <= N.range_value <= 2147483647", ["N"], "N不得小于1"),
            self._rel("1 <= H.range_value * rankSize.range_value <= 35000", ["H", "rankSize"], ""),
            self._rel("x.range_value <= 100", ["x"], "x不得超过100"),
            self._rel("y.range_value >= 1", ["y"], "y不得小于1"),
        ]
        lookup = _build_allowed_range_lookup(rels)
        for pname, (value, ar_type, _src) in lookup.items():
            if ar_type != "range":
                continue
            for item in value:
                if isinstance(item, list):
                    lo, hi = item[0], item[1]
                    assert lo is not None, f"{pname} has null lower bound"
                    assert hi is not None, f"{pname} has null upper bound"


class TestConstraintsInParametersArLookup:
    """Verify _build_constraints_in_parameters honors a caller-supplied
    ar_lookup (so implicit params get deduped consistently)."""

    def test_implicit_param_deduped_when_ar_complete(self):
        """When ar_lookup (filled, implicit-aware) marks BS as having a
        complete range, BS's self value_dependency expr is deduped from
        constraints_in_parameters (redundant with allowed_range_value)."""
        relations = [
            {"platform": "", "relation_object": {
                "expr_type": "value_dependency",
                "expr": "0 <= BS.range_value <= 2147483647",
                "relation_params": ["BS"],
                "src_text": "BS不得小于0",
            }},
            {"platform": "", "relation_object": {
                "expr_type": "shape_equality",
                "expr": "x1.shape[0] == BS",
                "relation_params": ["x1", "BS"],
                "src_text": "(BS, H)",
            }},
        ]
        supported = ["Atlas A2"]
        ar_lookup = {"BS": ([[0, 2147483647]], "range")}
        result = _build_constraints_in_parameters(
            relations, supported, [], ar_lookup=ar_lookup,
        )
        plat = "Atlas A2"
        # BS self value_dependency deduped; only shape_equality remains
        assert len(result[plat]) == 1
        assert result[plat][0]["expr_type"] == "shape_equality"

    def test_implicit_param_kept_when_ar_empty(self):
        """When BS has no allowed_range_value (incomplete/compound), its expr
        is retained in constraints_in_parameters."""
        relations = [
            {"platform": "", "relation_object": {
                "expr_type": "value_dependency",
                "expr": "1 <= H.range_value * rankSize.range_value <= 35000",
                "relation_params": ["H", "rankSize"],
                "src_text": "H*rankSize范围[1, 35000]",
            }},
        ]
        supported = ["Atlas A2"]
        # H has no AR entry (compound was filtered) → ar_lookup empty
        result = _build_constraints_in_parameters(relations, supported, [], ar_lookup={})
        plat = "Atlas A2"
        assert len(result[plat]) == 1
        assert result[plat][0]["expr"] == "1 <= H.range_value * rankSize.range_value <= 35000"

    def test_ar_lookup_none_falls_back_to_params(self):
        """Backward-compat: ar_lookup=None rebuilds from params (existing path)."""
        import json
        params = [{
            "param_name": "self",
            "param_constraint": json.dumps({
                "Atlas A2": {"allowed_range_value": {"value": [[0, 1]]}},
            }),
        }]
        relations = [
            {"platform": "", "relation_object": {
                "expr_type": "value_dependency",
                "expr": "0 <= self.range_value <= 1",
                "relation_params": ["self"],
                "src_text": "0~1",
            }},
        ]
        supported = ["Atlas A2"]
        result = _build_constraints_in_parameters(relations, supported, params)
        # self deduped → empty constraints list
        assert result.get("Atlas A2", []) == []


class TestBuildArLookupFromIo:
    def test_collects_filled_ar_including_implicit(self):
        """Inputs built from implicit params carry AR derived from relations."""
        mappings = [{
            "var_name": "BS", "tensor_param": "x1", "dim_index": 0,
            "shape_text": "(BS, H)", "is_constant": False, "is_external_constant": False,
        }]
        rels = [{"relation_object": {
            "expr_type": "value_dependency",
            "expr": "0 <= BS.range_value <= 2147483647",
            "relation_params": ["BS"],
            "src_text": "BS不得小于0，不得超过2147483647",
        }}]
        inputs, outputs = _build_inputs_outputs([], implicit_params=mappings, relations=rels)
        lookup = _build_ar_lookup_from_io(inputs, outputs)
        assert lookup["BS"] == ([[0, 2147483647]], "range")


class TestBuildInputsOutputsPrimaryFunction:
    """Single-function operator support: primary-function selection via signatures.

    Covers the three scenarios from the plan's verification matrix:
    (a) two-stage -> only GetWorkspaceSize params,
    (b) single-function -> all params,
    (c) empty/missing signatures -> legacy endswith fallback (no regression).
    """

    def test_two_stage_only_workspace_params(self):
        """Two-stage: only params from the GetWorkspaceSize function are kept."""
        signatures = [
            {"function_name": "aclnnAddGetWorkspaceSize"},
            {"function_name": "aclnnAdd"},
        ]
        params = [
            {"function_name": "aclnnAddGetWorkspaceSize", "param_name": "x",
             "direction": "input", "param_constraint": "{}"},
            {"function_name": "aclnnAddGetWorkspaceSize", "param_name": "y",
             "direction": "output", "param_constraint": "{}"},
            {"function_name": "aclnnAdd", "param_name": "workspaceAddr",
             "direction": "input", "param_constraint": "{}"},
        ]
        inputs, outputs = _build_inputs_outputs(params, signatures=signatures)
        assert "x" in inputs
        assert "y" in outputs
        # exec-only param must be excluded (only ws is primary)
        assert "workspaceAddr" not in inputs

    def test_single_function_all_params_included(self):
        """Single-function: the operator's only function is primary, all params kept."""
        signatures = [{"function_name": "aclnnCalculateMatmulWeightSize"}]
        params = [
            {"function_name": "aclnnCalculateMatmulWeightSize", "param_name": "tensorShape",
             "direction": "input", "param_constraint": "{}"},
            {"function_name": "aclnnCalculateMatmulWeightSize", "param_name": "weightTensorSize",
             "direction": "output", "param_constraint": "{}"},
        ]
        inputs, outputs = _build_inputs_outputs(params, signatures=signatures)
        assert "tensorShape" in inputs
        assert "weightTensorSize" in outputs

    def test_single_function_v2_all_params_included(self):
        """Single-function V2 (name does not end with WorkspaceSize)."""
        signatures = [{"function_name": "aclnnCalculateMatmulWeightSizeV2"}]
        params = [
            {"function_name": "aclnnCalculateMatmulWeightSizeV2", "param_name": "tensorShape",
             "direction": "input", "param_constraint": "{}"},
            {"function_name": "aclnnCalculateMatmulWeightSizeV2", "param_name": "dataType",
             "direction": "input", "param_constraint": "{}"},
            {"function_name": "aclnnCalculateMatmulWeightSizeV2", "param_name": "weightTensorSize",
             "direction": "output", "param_constraint": "{}"},
        ]
        inputs, outputs = _build_inputs_outputs(params, signatures=signatures)
        assert "tensorShape" in inputs
        assert "dataType" in inputs
        assert "weightTensorSize" in outputs

    def test_empty_signatures_falls_back_to_endswith(self):
        """Empty signatures -> legacy endswith fallback (no worse-than-status-quo)."""
        params = [
            {"function_name": "aclnnAddGetWorkspaceSize", "param_name": "x",
             "direction": "input", "param_constraint": "{}"},
            {"function_name": "aclnnAdd", "param_name": "workspaceAddr",
             "direction": "input", "param_constraint": "{}"},
        ]
        inputs, _ = _build_inputs_outputs(params, signatures=[])
        assert "x" in inputs
        # legacy endswith keeps only WorkspaceSize-ending functions
        assert "workspaceAddr" not in inputs

    def test_none_signatures_falls_back_to_endswith(self):
        """signatures=None (default) preserves legacy behavior for existing callers."""
        params = [
            {"function_name": "aclnnAddGetWorkspaceSize", "param_name": "x",
             "direction": "input", "param_constraint": "{}"},
            {"function_name": "aclnnAdd", "param_name": "workspaceAddr",
             "direction": "input", "param_constraint": "{}"},
        ]
        inputs, _ = _build_inputs_outputs(params)  # signatures defaults to None
        assert "x" in inputs
        assert "workspaceAddr" not in inputs

    def test_single_function_excluded_params_still_skipped(self):
        """EXCLUDED_PARAMS (workspace/workspaceSize/executor/stream) are skipped
        even for single-function operators (defensive — single-function sigs
        don't contain these, but the guard must hold)."""
        signatures = [{"function_name": "aclnnCalculateMatmulWeightSize"}]
        params = [
            {"function_name": "aclnnCalculateMatmulWeightSize", "param_name": "tensorShape",
             "direction": "input", "param_constraint": "{}"},
            {"function_name": "aclnnCalculateMatmulWeightSize", "param_name": "executor",
             "direction": "input", "param_constraint": "{}"},
        ]
        inputs, _ = _build_inputs_outputs(params, signatures=signatures)
        assert "tensorShape" in inputs
        assert "executor" not in inputs


class TestPrimaryFunctionHelpers:
    """Empty-input defense for get_primary_function_names / is_single_function_mode."""

    def test_get_primary_two_stage_returns_ws_only(self):
        sigs = [
            {"function_name": "aclnnAddGetWorkspaceSize"},
            {"function_name": "aclnnAdd"},
        ]
        assert get_primary_function_names(sigs) == {"aclnnAddGetWorkspaceSize"}

    def test_get_primary_single_function_returns_all(self):
        sigs = [{"function_name": "aclnnCalculateMatmulWeightSize"}]
        assert get_primary_function_names(sigs) == {"aclnnCalculateMatmulWeightSize"}

    def test_get_primary_empty_returns_none_sentinel(self):
        """Empty signatures -> None sentinel (caller falls back to endswith)."""
        assert get_primary_function_names([]) is None

    def test_is_single_function_mode_two_stage_false(self):
        sigs = [
            {"function_name": "aclnnAddGetWorkspaceSize"},
            {"function_name": "aclnnAdd"},
        ]
        assert is_single_function_mode(sigs) is False

    def test_is_single_function_mode_single_true(self):
        sigs = [{"function_name": "aclnnCalculateMatmulWeightSize"}]
        assert is_single_function_mode(sigs) is True

    def test_is_single_function_mode_empty_false(self):
        """Empty signatures -> False (must not misclassify as single-function)."""
        assert is_single_function_mode([]) is False

