"""Tests for the function_signature_extract node."""

from __future__ import annotations

from agent.nodes.function_signature_extract import (
    _normalize_param_types,
    _parse_json_response,
    _signatures_to_parameters,
)


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


class TestNormalizeParamTypes:
    def test_strips_const_modifier(self):
        sigs = [{"parameters": [{"name": "self", "type": "const aclTensor"}]}]
        result = _normalize_param_types(sigs)
        assert result[0]["parameters"][0]["type"] == "aclTensor"

    def test_strips_pointer_asterisk(self):
        sigs = [{"parameters": [{"name": "x", "type": "aclTensor*"}]}]
        result = _normalize_param_types(sigs)
        assert result[0]["parameters"][0]["type"] == "aclTensor"

    def test_strips_const_and_pointer(self):
        sigs = [{"parameters": [{"name": "self", "type": "const aclTensor *"}]}]
        result = _normalize_param_types(sigs)
        assert result[0]["parameters"][0]["type"] == "aclTensor"

    def test_strips_reference_operator(self):
        sigs = [{"parameters": [{"name": "x", "type": "int&"}]}]
        result = _normalize_param_types(sigs)
        assert result[0]["parameters"][0]["type"] == "int"

    def test_preserves_clean_type(self):
        sigs = [{"parameters": [{"name": "x", "type": "uint64_t"}]}]
        result = _normalize_param_types(sigs)
        assert result[0]["parameters"][0]["type"] == "uint64_t"

    def test_handles_empty_parameters(self):
        sigs = [{"parameters": []}]
        result = _normalize_param_types(sigs)
        assert result[0]["parameters"] == []

    def test_handles_missing_parameters_key(self):
        sigs = [{"function_name": "foo"}]
        result = _normalize_param_types(sigs)
        assert result == [{"function_name": "foo"}]

    def test_multiple_params_mixed(self):
        sigs = [{
            "parameters": [
                {"name": "self", "type": "const aclTensor *"},
                {"name": "stream", "type": "const aclrtStream"},
                {"name": "workspaceSize", "type": "uint64_t"},
            ]
        }]
        result = _normalize_param_types(sigs)
        params = result[0]["parameters"]
        assert params[0]["type"] == "aclTensor"
        assert params[1]["type"] == "aclrtStream"
        assert params[2]["type"] == "uint64_t"


class TestSignaturesToParameters:
    """Tests for _signatures_to_parameters flattening logic."""

    def test_basic_flattening(self):
        sigs = [{
            "function_name": "aclnnFoo",
            "parameters": [
                {"name": "x", "type": "aclTensor"},
                {"name": "y", "type": "int64_t"},
            ],
        }]
        result = _signatures_to_parameters(sigs)
        assert len(result) == 2
        assert result[0] == {"function_name": "aclnnFoo", "param_name": "x", "param_type": "aclTensor"}
        assert result[1] == {"function_name": "aclnnFoo", "param_name": "y", "param_type": "int64_t"}

    def test_no_direction_field(self):
        """direction should NOT be set — DB default 'input' takes effect."""
        sigs = [{
            "function_name": "aclnnFoo",
            "parameters": [{"name": "x", "type": "aclTensor"}],
        }]
        result = _signatures_to_parameters(sigs)
        assert "direction" not in result[0]

    def test_multiple_signatures(self):
        sigs = [
            {"function_name": "aclnnFooGetWorkspaceSize", "parameters": [{"name": "x", "type": "aclTensor"}]},
            {"function_name": "aclnnFoo", "parameters": [{"name": "x", "type": "aclTensor"}, {"name": "out", "type": "aclTensor"}]},
        ]
        result = _signatures_to_parameters(sigs)
        assert len(result) == 3
        assert result[0]["function_name"] == "aclnnFooGetWorkspaceSize"
        assert result[1]["function_name"] == "aclnnFoo"
        assert result[2]["function_name"] == "aclnnFoo"
        assert result[2]["param_name"] == "out"

    def test_empty_parameters(self):
        sigs = [{"function_name": "aclnnFoo", "parameters": []}]
        result = _signatures_to_parameters(sigs)
        assert result == []

    def test_empty_signatures(self):
        result = _signatures_to_parameters([])
        assert result == []

    def test_string_parameter_compat(self):
        """Handle legacy case where parameter is a plain string."""
        sigs = [{
            "function_name": "aclnnFoo",
            "parameters": ["x"],
        }]
        result = _signatures_to_parameters(sigs)
        assert len(result) == 1
        assert result[0]["param_name"] == "x"
        assert result[0]["param_type"] == ""

    def test_missing_type_field(self):
        sigs = [{
            "function_name": "aclnnFoo",
            "parameters": [{"name": "x"}],
        }]
        result = _signatures_to_parameters(sigs)
        assert result[0]["param_type"] == ""

    def test_normalized_type_preserved(self):
        """param_type should use the already-normalized type from signatures."""
        sigs = [{
            "function_name": "aclnnFoo",
            "parameters": [{"name": "x", "type": "aclTensor"}],  # already normalized
        }]
        result = _signatures_to_parameters(sigs)
        assert result[0]["param_type"] == "aclTensor"
