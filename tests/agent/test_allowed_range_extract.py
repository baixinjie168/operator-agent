"""Tests for the allowed_range_extract node."""

from __future__ import annotations

import json

from agent.utils.param_validators import (
    is_bool_type as _is_bool_type,
    is_tensor_type as _is_tensor_type,
    is_ws_function as _is_ws_function,
)
from agent.utils.llm_common import parse_json_response as _raw_parse

def _parse_allowed_range_response(text):
    return _raw_parse(text, list)


class TestIsWsFunction:
    def test_get_workspace_size_function(self):
        assert _is_ws_function("aclnnFooGetWorkspaceSize") is True

    def test_execute_function(self):
        assert _is_ws_function("aclnnFoo") is False

    def test_empty_function_name(self):
        assert _is_ws_function("") is False

    def test_partial_match(self):
        assert _is_ws_function("GetWorkspaceSize") is True


class TestParseAllowedRangeResponse:
    def test_valid_json_array(self):
        text = '[{"platform": "", "allowed_range_value": "取值范围为[0, 5120]"}]'
        result = _parse_allowed_range_response(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["platform"] == ""
        assert result[0]["allowed_range_value"] == "取值范围为[0, 5120]"

    def test_empty_array(self):
        result = _parse_allowed_range_response("[]")
        assert result is not None
        assert result == []

    def test_multiple_entries(self):
        text = json.dumps([
            {"platform": "Atlas A3", "allowed_range_value": "仅支持取值0"},
            {"platform": "", "allowed_range_value": "取值范围为[0, 100]"},
        ], ensure_ascii=False)
        result = _parse_allowed_range_response(text)
        assert result is not None
        assert len(result) == 2

    def test_json_in_code_block(self):
        text = '```json\n[{"platform": "", "allowed_range_value": "取值1或2"}]\n```'
        result = _parse_allowed_range_response(text)
        assert result is not None
        assert len(result) == 1

    def test_json_with_surrounding_text(self):
        text = '以下是结果：[{"platform": "", "allowed_range_value": "取值范围[1, 128]"}] 完毕'
        result = _parse_allowed_range_response(text)
        assert result is not None
        assert len(result) == 1

    def test_invalid_json_returns_none(self):
        result = _parse_allowed_range_response("这不是JSON")
        assert result is None

    def test_non_array_json_returns_none(self):
        result = _parse_allowed_range_response('{"key": "value"}')
        assert result is None

    def test_chinese_content(self):
        text = '[{"platform": "Atlas A2 训练系列产品", "allowed_range_value": "取值范围大于等于0"}]'
        result = _parse_allowed_range_response(text)
        assert result is not None
        assert result[0]["platform"] == "Atlas A2 训练系列产品"


class TestIsTensorType:
    def test_acl_tensor(self):
        assert _is_tensor_type("aclTensor") is True

    def test_acl_tensor_list(self):
        assert _is_tensor_type("aclTensorList") is True

    def test_const_acl_tensor(self):
        assert _is_tensor_type("const aclTensor") is True

    def test_int64_not_tensor(self):
        assert _is_tensor_type("int64_t") is False

    def test_bool_not_tensor(self):
        assert _is_tensor_type("bool") is False

    def test_acl_int_array_not_tensor(self):
        assert _is_tensor_type("aclIntArray") is False

    def test_empty_not_tensor(self):
        assert _is_tensor_type("") is False


class TestIsBoolType:
    def test_bool(self):
        assert _is_bool_type("bool") is True

    def test_bool_upper(self):
        assert _is_bool_type("Bool") is True

    def test_int_not_bool(self):
        assert _is_bool_type("int64_t") is False

    def test_tensor_not_bool(self):
        assert _is_bool_type("aclTensor") is False
