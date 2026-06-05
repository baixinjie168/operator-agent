"""Tests for the assemble_result node."""

from __future__ import annotations

from agent.nodes.assemble_result import (
    _build_function_explanation,
    _build_inputs_outputs,
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
            {"function_name": "aclnnFoo", "relation_type": "dtype", "description": "x1 and x2 must match"},
            {"function_name": "aclnnFoo", "relation_type": "shape", "description": "same shape"},
            {"function_name": "aclnnBar", "relation_type": "dtype", "description": "bar relation"},
        ]
        result = _build_function_explanation([], relations, [], [], [])
        assert len(result["aclnnFoo"]["relations"]) == 2
        assert len(result["aclnnBar"]["relations"]) == 1

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
        relations = [{"function_name": "aclnnFoo", "relation_type": "dtype"}]
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

