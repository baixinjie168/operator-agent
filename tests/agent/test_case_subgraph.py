"""Tests for the GeneratorAgent (case) sub-graph nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.nodes.case_subgraph import (
    case_generate_node,
    case_init_static_node,
    case_match_model_node,
    case_solve_constraints_node,
)


@pytest.fixture
def mock_mcp_client(monkeypatch):
    """Patch the module-level _mcp_client used by case nodes."""
    from agent.nodes.case_subgraph import match_model, generate

    client = MagicMock()
    client.get_json_constraints = AsyncMock()
    client.save_test_cases = AsyncMock()
    monkeypatch.setattr(match_model, "_mcp_client", client)
    monkeypatch.setattr(generate, "_mcp_client", client)
    return client


@pytest.mark.asyncio
class TestCaseMatchModel:
    async def test_happy_path(self, mock_mcp_client, sample_result_json) -> None:
        mock_mcp_client.get_json_constraints.return_value = sample_result_json
        result = await case_match_model_node({"operator_name": "aclnnSample"})
        assert result["error"] is None
        assert result["constraints_raw"] == sample_result_json

    async def test_missing_operator_name(self, mock_mcp_client) -> None:
        result = await case_match_model_node({})
        assert result["error"] is not None
        assert "operator_name" in result["error"]

    async def test_constraints_not_found(self, mock_mcp_client) -> None:
        mock_mcp_client.get_json_constraints.return_value = None
        result = await case_match_model_node({"operator_name": "aclnnX"})
        assert result["error"] is not None
        assert "json_constraints not found" in result["error"]
        assert result["constraints_raw"] is None


@pytest.mark.asyncio
class TestCaseInitStatic:
    async def test_happy_path(self, sample_result_json) -> None:
        result = await case_init_static_node({"constraints_raw": sample_result_json})
        assert result["error"] is None
        assert result["sampled_shapes"] >= 1
        assert result["sampled_dtypes"] >= 1

    async def test_empty_constraints(self) -> None:
        result = await case_init_static_node({"constraints_raw": {"inputs": {}, "outputs": {}, "dtype_support_description": {}}})
        assert result["error"] is None
        assert result["sampled_shapes"] == 0
        assert result["sampled_dtypes"] == 0


@pytest.mark.asyncio
class TestCaseSolveConstraints:
    async def test_happy_path(self, sample_result_json) -> None:
        result = await case_solve_constraints_node({
            "constraints_raw": sample_result_json,
            "sampled_shapes": 10,
            "sampled_dtypes": 4,
        })
        assert result["error"] is None
        assert result["valid_combos"] == 24  # 10*4*0.6
        assert result["rejected_combos"] == 16

    async def test_zero_candidates(self, sample_result_json) -> None:
        result = await case_solve_constraints_node({
            "constraints_raw": sample_result_json,
            "sampled_shapes": 0,
            "sampled_dtypes": 0,
        })
        assert result["valid_combos"] == 0
        assert result["rejected_combos"] == 0


@pytest.mark.asyncio
class TestCaseGenerate:
    async def test_happy_path(
        self, mock_mcp_client, sample_result_json,
    ) -> None:
        mock_mcp_client.save_test_cases.return_value = {
            "operator_name": "aclnnSample",
            "saved_count": 5,
            "output_path": "/tmp/cases/aclnnSample_cases.json",
        }
        result = await case_generate_node({
            "operator_name": "aclnnSample",
            "constraints_raw": sample_result_json,
            "cases_count": 5,
            "cases_seed": 7,
        })
        assert result["error"] is None
        assert result["cases_count"] == 5
        assert result["cases_path"] == "/tmp/cases/aclnnSample_cases.json"
        mock_mcp_client.save_test_cases.assert_awaited_once()

    async def test_skipped_when_error(self) -> None:
        result = await case_generate_node({"error": "no constraints"})
        assert result["error"] == "no constraints"
        assert result.get("cases_path") is None
        assert result.get("cases_count") is None

    async def test_missing_inputs(self) -> None:
        result = await case_generate_node({})
        assert result["error"] is not None

    async def test_default_count_and_seed(
        self, mock_mcp_client, sample_result_json,
    ) -> None:
        mock_mcp_client.save_test_cases.return_value = {
            "operator_name": "aclnnSample",
            "saved_count": 10,
            "output_path": "/tmp/cases/aclnnSample_cases.json",
        }
        result = await case_generate_node({
            "operator_name": "aclnnSample",
            "constraints_raw": sample_result_json,
        })
        # Default count=10, seed=42 — just verify it ran with defaults
        assert result["error"] is None
        assert result["cases_count"] >= 1


def test_case_subgraph_compiles() -> None:
    """The sub-graph builder must produce a CompiledStateGraph."""
    from agent.nodes.case_subgraph import create_case_subgraph

    sg = create_case_subgraph()
    assert sg.name == "case-subgraph"
