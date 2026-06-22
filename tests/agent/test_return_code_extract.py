"""Tests for the return_code_extract node."""

from __future__ import annotations

import json

from agent.utils.llm_common import parse_json_response as _raw_parse

def _parse_json_response(text):
    return _raw_parse(text, list)


class TestParseJsonResponse:
    def test_valid_json_array(self):
        text = json.dumps([
            {
                "return_value": "ACLNN_ERR_PARAM_NULLPTR",
                "error_code": 161001,
                "descriptions": ["传入的指针是空指针"],
            }
        ], ensure_ascii=False)
        result = _parse_json_response(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["return_value"] == "ACLNN_ERR_PARAM_NULLPTR"
        assert result[0]["error_code"] == 161001
        assert result[0]["descriptions"] == ["传入的指针是空指针"]

    def test_empty_array(self):
        result = _parse_json_response("[]")
        assert result is not None
        assert result == []

    def test_multiple_entries(self):
        text = json.dumps([
            {
                "return_value": "ACLNN_ERR_PARAM_NULLPTR",
                "error_code": 161001,
                "descriptions": ["描述1", "描述2"],
            },
            {
                "return_value": "ACLNN_ERR_PARAM_INVALID",
                "error_code": 161002,
                "descriptions": ["数据类型不在支持的范围之内"],
            },
        ], ensure_ascii=False)
        result = _parse_json_response(text)
        assert result is not None
        assert len(result) == 2
        assert result[1]["error_code"] == 161002

    def test_json_in_code_block(self):
        text = '```json\n[{"return_value": "ACLNN_ERR_PARAM_NULLPTR", "error_code": 161001, "descriptions": ["空指针"]}]\n```'
        result = _parse_json_response(text)
        assert result is not None
        assert len(result) == 1

    def test_json_with_surrounding_text(self):
        text = '以下是提取结果：[{"return_value": "ACLNN_ERR_PARAM_NULLPTR", "error_code": 161001, "descriptions": ["空指针"]}] 完毕'
        result = _parse_json_response(text)
        assert result is not None
        assert len(result) == 1

    def test_invalid_json_returns_none(self):
        result = _parse_json_response("这不是 JSON")
        assert result is None

    def test_bare_json_array(self):
        text = '[{"return_value": "ACLNN_ERR_PARAM_NULLPTR", "error_code": 161001, "descriptions": ["空指针"]}]'
        result = _parse_json_response(text)
        assert result is not None
        assert len(result) == 1

    def test_error_code_is_integer(self):
        text = '[{"return_value": "ACLNN_ERR_PARAM_NULLPTR", "error_code": 161001, "descriptions": ["空指针"]}]'
        result = _parse_json_response(text)
        assert result is not None
        assert isinstance(result[0]["error_code"], int)

