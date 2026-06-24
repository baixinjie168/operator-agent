"""Tests for implicit parameter Agent validation (Phase 2).

Covers:
- _parse_agent_response: JSON parsing from Agent text output
- _candidate_to_mapping: candidate-to-mapping conversion
- _apply_agent_actions: confirm/remove/reclassify/additions logic
- Fallback: Agent failure returns regex candidates
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
for pkg in ("packages/shared/src", "packages/mcp-server/src", "packages/agent/src"):
    p = str(ROOT / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.nodes.param_relation_extract.implicit_param_extract import (  # noqa: E402
    _apply_agent_actions,
    _build_agent_user_message,
    _candidate_to_mapping,
    _parse_agent_response,
    _validate_via_agent,
)


def _make_candidate(
    cid: str = "cand_001",
    var_name: str = "BS",
    tensor_param: str = "x1",
    dim_index: int = 0,
) -> dict:
    return {
        "candidate_id": cid,
        "var_name": var_name,
        "tensor_param": tensor_param,
        "dim_index": dim_index,
        "slot_index": dim_index,
        "slot_expr": var_name,
        "is_compound": False,
        "compound_expr": None,
        "shape_text": f"({var_name})",
        "surrounding_text": f"...shape of {tensor_param} is ({var_name})...",
    }


# ---------------------------------------------------------------------------
# _parse_agent_response
# ---------------------------------------------------------------------------


class TestParseAgentResponse:
    def test_valid_json(self):
        result = _parse_agent_response('{"actions": [], "additions": []}')
        assert result is not None
        assert result["actions"] == []

    def test_json_in_code_block(self):
        text = '```json\n{"actions": [], "additions": []}\n```'
        result = _parse_agent_response(text)
        assert result is not None
        assert result["actions"] == []

    def test_garbage_returns_none(self):
        assert _parse_agent_response("not json at all") is None

    def test_empty_string_returns_none(self):
        assert _parse_agent_response("") is None

    def test_partial_json_extracted(self):
        text = 'Result: {"actions": [], "additions": []} done.'
        result = _parse_agent_response(text)
        assert result is not None
        assert result["actions"] == []


# ---------------------------------------------------------------------------
# _candidate_to_mapping
# ---------------------------------------------------------------------------


class TestCandidateToMapping:
    def test_basic_conversion(self):
        c = _make_candidate(var_name="BS", tensor_param="x1", dim_index=0)
        m = _candidate_to_mapping(c)
        assert m["var_name"] == "BS"
        assert m["tensor_param"] == "x1"
        assert m["dim_index"] == 0
        assert m["is_constant"] is False
        assert m["is_external_constant"] is False
        assert m["is_quantization_type"] is False
        assert m["referenced_in"] == []

    def test_compound_candidate(self):
        c = _make_candidate(var_name="H", dim_index=0)
        c["is_compound"] = True
        c["compound_expr"] = "H*rankSize"
        m = _candidate_to_mapping(c)
        assert m["is_compound"] is True
        assert m["compound_expr"] == "H*rankSize"


# ---------------------------------------------------------------------------
# _apply_agent_actions
# ---------------------------------------------------------------------------


class TestApplyAgentActions:
    def test_confirm_action(self):
        candidates = [_make_candidate(cid="cand_001", var_name="BS")]
        parsed = {
            "actions": [
                {"candidate_id": "cand_001", "action": "confirm",
                 "classification": "dimension_variable", "reason": "dim var"}
            ],
            "additions": [],
        }
        mappings = _apply_agent_actions(candidates, parsed)
        assert len(mappings) == 1
        assert mappings[0]["var_name"] == "BS"

    def test_remove_action(self):
        candidates = [
            _make_candidate(cid="cand_001", var_name="Reduce"),
            _make_candidate(cid="cand_002", var_name="BS"),
        ]
        parsed = {
            "actions": [
                {"candidate_id": "cand_001", "action": "remove",
                 "reason": "concept term"},
                {"candidate_id": "cand_002", "action": "confirm",
                 "classification": "dimension_variable", "reason": "dim var"},
            ],
            "additions": [],
        }
        mappings = _apply_agent_actions(candidates, parsed)
        assert len(mappings) == 1
        assert mappings[0]["var_name"] == "BS"

    def test_reclassify_to_constant(self):
        candidates = [_make_candidate(cid="cand_001", var_name="k0")]
        parsed = {
            "actions": [
                {"candidate_id": "cand_001", "action": "reclassify",
                 "classification": "constant", "constant_value": 16,
                 "reason": "k0 = 16 in text"}
            ],
            "additions": [],
        }
        mappings = _apply_agent_actions(candidates, parsed)
        assert len(mappings) == 1
        assert mappings[0]["var_name"] == "k0"
        assert mappings[0]["is_constant"] is True
        assert mappings[0]["constant_value"] == 16

    def test_reclassify_to_external_constant(self):
        candidates = [_make_candidate(cid="cand_001", var_name="rankSize")]
        parsed = {
            "actions": [
                {"candidate_id": "cand_001", "action": "reclassify",
                 "classification": "external_constant",
                 "referenced_in": ["x1"],
                 "reason": "only in compound expr"}
            ],
            "additions": [],
        }
        mappings = _apply_agent_actions(candidates, parsed)
        assert len(mappings) == 1
        assert mappings[0]["var_name"] == "rankSize"
        assert mappings[0]["is_external_constant"] is True
        assert mappings[0]["tensor_param"] is None
        assert mappings[0]["referenced_in"] == ["x1"]

    def test_additions(self):
        candidates = [_make_candidate(cid="cand_001", var_name="N")]
        parsed = {
            "actions": [
                {"candidate_id": "cand_001", "action": "confirm",
                 "classification": "dimension_variable", "reason": "dim var"}
            ],
            "additions": [
                {"var_name": "groupSize",
                 "classification": "external_constant",
                 "referenced_in": ["x1"],
                 "reason": "found in constraint text"}
            ],
        }
        mappings = _apply_agent_actions(candidates, parsed)
        assert len(mappings) == 2
        assert mappings[0]["var_name"] == "N"
        assert mappings[1]["var_name"] == "groupSize"
        assert mappings[1]["is_external_constant"] is True

    def test_unhandled_candidates_kept(self):
        """Candidates not mentioned by Agent should be kept as fallback."""
        candidates = [
            _make_candidate(cid="cand_001", var_name="BS"),
            _make_candidate(cid="cand_002", var_name="H"),
        ]
        parsed = {
            "actions": [
                {"candidate_id": "cand_001", "action": "confirm",
                 "classification": "dimension_variable", "reason": "ok"}
            ],
            "additions": [],
        }
        mappings = _apply_agent_actions(candidates, parsed)
        assert len(mappings) == 2
        var_names = {m["var_name"] for m in mappings}
        assert var_names == {"BS", "H"}


# ---------------------------------------------------------------------------
# _build_agent_user_message
# ---------------------------------------------------------------------------


class TestBuildAgentUserMessage:
    def test_contains_all_fields(self):
        msg = _build_agent_user_message(
            [_make_candidate()], "section text", {"x1"}, {"x1"}
        )
        data = json.loads(msg)
        assert "candidates" in data
        assert "section_text" in data
        assert "signature_params" in data
        assert "tensor_params" in data
        assert data["section_text"] == "section text"


# ---------------------------------------------------------------------------
# _validate_via_agent (async, with mocks)
# ---------------------------------------------------------------------------


class TestValidateViaAgent:
    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self):
        result = await _validate_via_agent([], "text", set(), set())
        assert result == []

    @pytest.mark.asyncio
    async def test_agent_failure_falls_back(self):
        """When Agent throws, should return regex candidates as-is."""
        candidates = [_make_candidate(var_name="BS")]
        with patch(
            "agent.nodes.param_relation_extract.implicit_param_extract."
            "_get_implicit_params_agent"
        ) as mock_get:
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("timeout"))
            mock_agent._implicit_params_system_prompt = "sys"
            mock_get.return_value = mock_agent
            result = await _validate_via_agent(candidates, "text", set(), set())
        assert len(result) == 1
        assert result[0]["var_name"] == "BS"

    @pytest.mark.asyncio
    async def test_agent_unparseable_falls_back(self):
        """When Agent returns garbage, should return regex candidates."""
        candidates = [_make_candidate(var_name="BS")]
        with patch(
            "agent.nodes.param_relation_extract.implicit_param_extract."
            "_get_implicit_params_agent"
        ) as mock_get:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "not json at all"
            mock_agent.ainvoke = AsyncMock(return_value=mock_response)
            mock_agent._implicit_params_system_prompt = "sys"
            mock_get.return_value = mock_agent
            result = await _validate_via_agent(candidates, "text", set(), set())
        assert len(result) == 1
        assert result[0]["var_name"] == "BS"

    @pytest.mark.asyncio
    async def test_agent_removes_concept_term(self):
        """Agent should remove Reduce and keep BS."""
        candidates = [
            _make_candidate(cid="cand_001", var_name="Reduce"),
            _make_candidate(cid="cand_002", var_name="BS"),
        ]
        agent_response = json.dumps({
            "actions": [
                {"candidate_id": "cand_001", "action": "remove",
                 "reason": "concept term"},
                {"candidate_id": "cand_002", "action": "confirm",
                 "classification": "dimension_variable", "reason": "dim var"}
            ],
            "additions": []
        })
        with patch(
            "agent.nodes.param_relation_extract.implicit_param_extract."
            "_get_implicit_params_agent"
        ) as mock_get:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.content = agent_response
            mock_agent.ainvoke = AsyncMock(return_value=mock_response)
            mock_agent._implicit_params_system_prompt = "sys"
            mock_get.return_value = mock_agent
            result = await _validate_via_agent(candidates, "text", set(), set())
        assert len(result) == 1
        assert result[0]["var_name"] == "BS"