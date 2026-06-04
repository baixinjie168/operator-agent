"""Tests for the assemble_result node."""

from __future__ import annotations

from agent.nodes.assemble_result import _build_function_explanation


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

