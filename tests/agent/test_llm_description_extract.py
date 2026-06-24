"""Tests for the llm_description_extract node."""

from __future__ import annotations

import json

from agent.nodes.context_utils import _is_ws_function
from agent.nodes.llm_description_extract.extract_ws import (
    _build_discontinuous_json,
    _parse_direction,
    _parse_llm_response,
)
from agent.nodes.llm_description_extract.save_descriptions import (
    _build_enriched_params,
)
from agent.prompts import LLM_DESCRIPTION_EXTRACT_PROMPT


class TestIsWsFunction:
    def test_get_workspace_size(self):
        assert _is_ws_function("aclnnFooGetWorkspaceSize") is True

    def test_execute_function(self):
        assert _is_ws_function("aclnnFoo") is False

    def test_empty(self):
        assert _is_ws_function("") is False


class TestParseDirection:
    def test_input_chinese(self):
        assert _parse_direction("输入") == "input"

    def test_output_chinese(self):
        assert _parse_direction("输出") == "output"

    def test_input_with_noise(self):
        assert _parse_direction("**输入**") == "input"

    def test_unknown(self):
        assert _parse_direction("未知") == ""

    def test_empty(self):
        assert _parse_direction("") == ""

    def test_input_longer_text(self):
        assert _parse_direction("输入（input）") == "input"

    def test_output_compute(self):
        assert _parse_direction("计算输出") == "output"


class TestBuildDiscontinuousJson:
    def test_non_tensor_returns_na(self):
        result = json.loads(_build_discontinuous_json(True, "int64_t"))
        assert result["value"] == "N/A"

    def test_tensor_true(self):
        result = json.loads(_build_discontinuous_json(True, "aclTensor"))
        assert result["value"] is True

    def test_tensor_false(self):
        result = json.loads(_build_discontinuous_json(False, "aclTensor"))
        assert result["value"] is False

    def test_tensor_null(self):
        result = json.loads(_build_discontinuous_json(None, "aclTensor"))
        assert result["value"] is False

    def test_non_tensor_case_insensitive(self):
        result = json.loads(_build_discontinuous_json(True, "ACLTensor"))
        assert result["value"] is True


class TestParseLlmResponse:
    def test_valid_json(self):
        text = '{"llm_description": "desc", "src_content": "src", "direction": "输入"}'
        result = _parse_llm_response(text)
        assert result is not None
        assert result["llm_description"] == "desc"

    def test_json_in_code_block(self):
        text = '```json\n{"llm_description": "desc"}\n```'
        result = _parse_llm_response(text)
        assert result is not None
        assert result["llm_description"] == "desc"

    def test_invalid_returns_none(self):
        result = _parse_llm_response("not valid json at all")
        assert result is None


class TestBuildEnrichedParams:
    def test_merge_updates(self):
        original = [
            {"function_name": "fn", "param_name": "x", "param_type": "aclTensor"},
        ]
        updates = [
            {
                "function_name": "fn",
                "param_name": "x",
                "llm_description": "desc",
                "src_content": "src",
                "direction": "input",
                "is_support_discontinuous": '{"value": false, "src_text": ""}',
            },
        ]
        result = _build_enriched_params(original, updates)
        assert len(result) == 1
        assert result[0]["llm_description"] == "desc"
        assert result[0]["src_content"] == "src"
        assert result[0]["direction"] == "input"
        assert result[0]["param_type"] == "aclTensor"

    def test_no_update_keeps_original(self):
        original = [
            {"function_name": "fn", "param_name": "x", "param_type": "aclTensor"},
        ]
        result = _build_enriched_params(original, [])
        assert len(result) == 1
        assert "llm_description" not in result[0]

    def test_partial_update(self):
        original = [
            {"function_name": "fn", "param_name": "x"},
            {"function_name": "fn", "param_name": "y"},
        ]
        updates = [
            {"function_name": "fn", "param_name": "x", "llm_description": "desc x",
             "src_content": "", "direction": "input",
             "is_support_discontinuous": '{"value": "N/A", "src_text": ""}'},
        ]
        result = _build_enriched_params(original, updates)
        assert len(result) == 2
        assert result[0]["llm_description"] == "desc x"
        assert "llm_description" not in result[1]


class TestPromptExcludesCrossParamRelations:
    """Verify prompt rule 7 excludes cross-parameter relationship constraints."""

    def test_prompt_contains_rule_7(self):
        assert "不要包含**与其他参数之间的关系约束" in LLM_DESCRIPTION_EXTRACT_PROMPT

    def test_prompt_examples_include_dtype_relation(self):
        assert "与参数X的数据类型一致" in LLM_DESCRIPTION_EXTRACT_PROMPT

    def test_prompt_examples_include_shape_relation(self):
        assert "shape的第N维与参数Y相同" in LLM_DESCRIPTION_EXTRACT_PROMPT

    def test_prompt_examples_include_value_dependency(self):
        assert "取值依赖参数Z" in LLM_DESCRIPTION_EXTRACT_PROMPT

    def test_prompt_examples_include_input_dtype(self):
        assert "必须与input的dtype保持一致" in LLM_DESCRIPTION_EXTRACT_PROMPT

    def test_prompt_mentions_other_modules(self):
        assert "这类跨参数关系由其他模块专门处理" in LLM_DESCRIPTION_EXTRACT_PROMPT

