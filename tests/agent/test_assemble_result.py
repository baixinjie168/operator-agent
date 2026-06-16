"""Tests for the assemble_result node."""

from __future__ import annotations

from agent.nodes.assemble_result import (
    _build_constraints_in_parameters,
    _build_function_explanation,
    _build_inputs_outputs,
    _has_meaningful_expr,
    _transform_return_codes,
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
            {"function_name": "aclnnFoo", "platform": "通用", "combo": {"x1": "FLOAT16"}},
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
