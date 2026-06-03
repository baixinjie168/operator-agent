"""Tests for the function_signature_extract node."""

from __future__ import annotations

from agent.nodes.function_signature_extract import _parse_json_response


class TestParseJsonResponse:
    def test_parses_plain_json_array(self):
        text = '[{"function_name": "foo", "return_type": "int"}]'
        result = _parse_json_response(text)
        assert len(result) == 1
        assert result[0]["function_name"] == "foo"

    def test_parses_json_in_code_block(self):
        text = '```json\n[{"function_name": "bar"}]\n```'
        result = _parse_json_response(text)
        assert len(result) == 1
        assert result[0]["function_name"] == "bar"

    def test_parses_json_in_plain_code_block(self):
        text = '```\n[{"function_name": "baz"}]\n```'
        result = _parse_json_response(text)
        assert len(result) == 1
        assert result[0]["function_name"] == "baz"

    def test_parses_json_with_surrounding_text(self):
        text = 'Here is the result:\n[{"function_name": "test"}]\nDone.'
        result = _parse_json_response(text)
        assert len(result) == 1
        assert result[0]["function_name"] == "test"

    def test_returns_empty_for_invalid_json(self):
        text = 'not json at all'
        result = _parse_json_response(text)
        assert result == []

    def test_returns_empty_for_non_array(self):
        text = '{"function_name": "foo"}'
        result = _parse_json_response(text)
        assert result == []

    def test_parses_multiple_signatures(self):
        text = '''[
          {"function_name": "foo", "return_type": "int", "parameters": [{"name": "x", "type": "int"}]},
          {"function_name": "bar", "return_type": "void", "parameters": []}
        ]'''
        result = _parse_json_response(text)
        assert len(result) == 2
        assert result[0]["function_name"] == "foo"
        assert result[1]["function_name"] == "bar"
        assert len(result[0]["parameters"]) == 1
        assert result[0]["parameters"][0]["name"] == "x"
