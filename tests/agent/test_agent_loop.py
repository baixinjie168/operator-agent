"""Tests for agent_loop.py: extract_relations_agent + _cleanup."""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agent.nodes.param_relation_extract.agent_loop import (
    _cleanup,
    extract_relations_agent,
)


class TestCleanup:
    def test_removes_source_markers(self):
        relations = [
            {"params": ["x"], "_source": "targeted:x", "_source_chunk": 0},
            {"params": ["y"], "_source": "self_check"},
        ]
        result = _cleanup(relations)
        for r in result:
            assert "_source" not in r
            assert "_source_chunk" not in r

    def test_preserves_relation_fields(self):
        relations = [
            {
                "relation_type": "shape",
                "params": ["x", "y"],
                "description": "test",
                "_source": "test",
            },
        ]
        result = _cleanup(relations)
        assert result[0]["relation_type"] == "shape"
        assert result[0]["params"] == ["x", "y"]

    def test_empty_list(self):
        assert _cleanup([]) == []


class TestExtractRelationsAgent:
    @pytest.mark.asyncio
    async def test_empty_content_returns_empty(self):
        llm = MagicMock()
        relations, report = await extract_relations_agent("", ["x"], llm)
        assert relations == []
        assert report["total"] == 0
        assert report["total_rounds"] == 1

    @pytest.mark.asyncio
    async def test_empty_params_returns_early(self):
        llm = MagicMock()
        relations, report = await extract_relations_agent("some content", [], llm)
        assert report["total_rounds"] == 1

    @pytest.mark.asyncio
    async def test_report_fields_present(self):
        llm = MagicMock()
        _, report = await extract_relations_agent("", [], llm)
        expected_keys = {"round1", "round3", "round4", "self_check_rounds",
                         "total_rounds", "total", "uncovered_params", "coverage"}
        assert expected_keys.issubset(set(report.keys()))

    @pytest.mark.asyncio
    async def test_exception_fallback(self):
        """Agent loop should return partial results on exception."""
        llm = MagicMock()
        # Make the LLM raise an exception
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM failure"))

        relations, report = await extract_relations_agent(
            "some content with x and y", ["x", "y"], llm
        )
        # Should not raise, should return whatever we have (possibly empty)
        assert isinstance(relations, list)
        assert isinstance(report, dict)
        assert "total" in report

    @pytest.mark.asyncio
    async def test_timeout_returns_round1_results(self):
        """Agent loop should return round-1 results when wall-time limit is exceeded."""
        llm = MagicMock()

        async def slow_ainvoke(prompt):
            await asyncio.sleep(2)
            resp = MagicMock()
            resp.content = "[]"
            return resp

        llm.ainvoke = AsyncMock(side_effect=slow_ainvoke)

        # Patch timeout to 0.5s so the first chunked call itself times out
        with patch(
            "agent.nodes.param_relation_extract.agent_loop.MAX_WALL_TIME_PER_DOC",
            0.5,
        ):
            relations, report = await extract_relations_agent(
                "some content mentioning x and y", ["x", "y"], llm,
            )

        assert isinstance(relations, list)
        assert report.get("timed_out") is True
        assert report["total_rounds"] == 1
