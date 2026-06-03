"""Tests for the determinism_extract node."""

from __future__ import annotations

from agent.nodes.determinism_extract import _expand_platforms, _parse_llm_response


class TestParseLlmResponse:
    def test_valid_json_array(self):
        text = '[{"product": "", "value": true, "src_text": "aclnnAdaLayerNorm默认确定性实现。"}]'
        result = _parse_llm_response(text)
        assert len(result) == 1
        assert result[0]["product"] == ""
        assert result[0]["value"] is True
        assert "确定性" in result[0]["src_text"]

    def test_empty_array(self):
        result = _parse_llm_response("[]")
        assert result == []

    def test_multiple_entries(self):
        text = '[{"product": "Atlas A2", "value": true, "src_text": "src1"}, {"product": "Atlas A3", "value": false, "src_text": "src2"}]'
        result = _parse_llm_response(text)
        assert len(result) == 2
        assert result[0]["product"] == "Atlas A2"
        assert result[0]["value"] is True
        assert result[1]["product"] == "Atlas A3"
        assert result[1]["value"] is False

    def test_json_in_code_block(self):
        text = '```json\n[{"product": "", "value": true, "src_text": "test"}]\n```'
        result = _parse_llm_response(text)
        assert len(result) == 1
        assert result[0]["value"] is True

    def test_invalid_json_returns_empty(self):
        result = _parse_llm_response("这不是 JSON")
        assert result == []

    def test_value_normalization(self):
        text = '[{"product": "", "value": 1, "src_text": "test"}]'
        result = _parse_llm_response(text)
        assert len(result) == 1
        assert result[0]["value"] is True

    def test_missing_fields(self):
        text = '[{"value": true}]'
        result = _parse_llm_response(text)
        assert len(result) == 1
        assert result[0]["product"] == ""
        assert result[0]["src_text"] == ""


class TestExpandPlatforms:
    def test_empty_product_expands_to_all_platforms(self):
        records = [{"product": "", "value": True, "src_text": "test"}]
        platforms = ["Atlas A2", "Atlas A3"]
        result = _expand_platforms(records, platforms)
        assert len(result) == 2
        assert result[0]["product"] == "Atlas A2"
        assert result[0]["value"] is True
        assert result[1]["product"] == "Atlas A3"
        assert result[1]["value"] is True

    def test_specific_product_kept_as_is(self):
        records = [{"product": "Atlas A2", "value": False, "src_text": "test"}]
        platforms = ["Atlas A2", "Atlas A3"]
        result = _expand_platforms(records, platforms)
        assert len(result) == 1
        assert result[0]["product"] == "Atlas A2"
        assert result[0]["value"] is False

    def test_mixed_records(self):
        records = [
            {"product": "", "value": True, "src_text": "src1"},
            {"product": "Atlas A2", "value": False, "src_text": "src2"},
        ]
        platforms = ["Atlas A2", "Atlas A3"]
        result = _expand_platforms(records, platforms)
        assert len(result) == 3
        assert result[0]["product"] == "Atlas A2"
        assert result[0]["value"] is True
        assert result[1]["product"] == "Atlas A3"
        assert result[1]["value"] is True
        assert result[2]["product"] == "Atlas A2"
        assert result[2]["value"] is False

    def test_no_supported_platforms(self):
        records = [{"product": "", "value": True, "src_text": "test"}]
        platforms = []
        result = _expand_platforms(records, platforms)
        assert len(result) == 0
