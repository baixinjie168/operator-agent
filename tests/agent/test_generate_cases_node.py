"""Tests for agent.nodes.generate_cases."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.nodes.generate_cases import generate_cases_node


@pytest.fixture
def mock_mcp_client(monkeypatch):
    """Patch the module-level _mcp_client used by the node."""
    from agent.nodes import generate_cases

    client = MagicMock()
    client.get_json_constraints = AsyncMock()
    client.save_test_cases = AsyncMock()
    monkeypatch.setattr(generate_cases, "_mcp_client", client)
    return client


@pytest.mark.asyncio
class TestGenerateCasesNode:
    async def test_happy_path(self, mock_mcp_client, sample_result_json: dict) -> None:
        mock_mcp_client.get_json_constraints.return_value = sample_result_json
        mock_mcp_client.save_test_cases.return_value = {
            "operator_name": "aclnnSample",
            "saved_count": 10,
            "output_path": "/tmp/cases/aclnnSample_cases.json",
        }
        result = await generate_cases_node({
            "operator_name": "aclnnSample",
        })
        assert result["error"] is None
        assert result["cases_count"] == 10
        assert result["cases_path"] == "/tmp/cases/aclnnSample_cases.json"
        mock_mcp_client.save_test_cases.assert_awaited_once()
        # Verify the saved cases are valid JSON list
        call_args = mock_mcp_client.save_test_cases.call_args
        cases_json = call_args.kwargs["cases_json"]
        import json
        cases = json.loads(cases_json)
        assert isinstance(cases, list)
        assert len(cases) == 10

    async def test_missing_operator_name(self, mock_mcp_client) -> None:
        result = await generate_cases_node({})
        assert result["error"] is not None
        assert "operator_name" in result["error"]
        assert result["cases_path"] is None
        mock_mcp_client.get_json_constraints.assert_not_awaited()

    async def test_no_constraints_found(self, mock_mcp_client) -> None:
        mock_mcp_client.get_json_constraints.return_value = None
        result = await generate_cases_node({"operator_name": "aclnnMissing"})
        assert result["error"] is not None
        assert "json_constraints" in result["error"]
        assert result["cases_path"] is None

    async def test_exception_bubbles_as_error(self, mock_mcp_client, sample_result_json: dict) -> None:
        mock_mcp_client.get_json_constraints.return_value = sample_result_json
        mock_mcp_client.save_test_cases.side_effect = RuntimeError("mcp down")
        result = await generate_cases_node({"operator_name": "aclnnSample"})
        assert result["error"] == "mcp down"
        assert result["cases_path"] is None
        assert result["cases_count"] is None

    async def test_custom_count_and_seed(self, mock_mcp_client, sample_result_json: dict) -> None:
        mock_mcp_client.get_json_constraints.return_value = sample_result_json
        mock_mcp_client.save_test_cases.return_value = {
            "operator_name": "aclnnSample",
            "saved_count": 3,
            "output_path": "/tmp/cases/aclnnSample_cases.json",
        }
        result = await generate_cases_node({
            "operator_name": "aclnnSample",
            "cases_count": 3,
            "cases_seed": 7,
        })
        assert result["cases_count"] == 3
        call_args = mock_mcp_client.save_test_cases.call_args
        import json
        cases = json.loads(call_args.kwargs["cases_json"])
        assert len(cases) == 3
