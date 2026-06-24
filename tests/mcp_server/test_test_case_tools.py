"""Tests for mcp_server.tools.test_case_tools."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_cases_dir(tmp_path: Path, monkeypatch) -> Path:
    """Force the on-disk cases dir to a tmp_path under the project root layout.

    We monkey-patch ``_resolve_output_path`` indirectly by setting an env var
    that the tool reads (TODO if needed) — for now we use a tmp path and
    patch the module's CASES_DIR_NAME resolution.
    """
    return tmp_path


def _patched_save(operator_name: str, cases_json: str, tmp_path: Path):
    """Run do_save_test_cases with a tmp output dir."""
    from mcp_server.tools import test_case_tools

    # Patch the helper to point at tmp_path.
    def fake_resolve(op_name: str, output_dir: str | None) -> Path:
        base = Path(output_dir) if output_dir else tmp_path
        return base / f"{op_name}_cases.json"

    monkey = pytest.MonkeyPatch()
    monkey.setattr(test_case_tools, "_resolve_output_path", fake_resolve)
    try:
        return test_case_tools.do_save_test_cases(operator_name, cases_json)
    finally:
        monkey.undo()


class TestDoSaveTestCases:
    def test_saves_to_db_and_disk(self, tmp_path: Path) -> None:
        from mcp_server.tools import test_case_tools

        test_case_tools.ensure_test_cases_schema()
        cases = [{"id": 0, "name": "aclnnX"}, {"id": 1, "name": "aclnnX"}]
        result = _patched_save(
            "aclnnX", json.dumps(cases), tmp_path,
        )
        assert result["saved_count"] == 2
        assert result["operator_name"] == "aclnnX"
        out_path = Path(result["output_path"])
        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded == cases

    def test_invalid_json_raises(self) -> None:
        from mcp_server.tools import test_case_tools

        with pytest.raises(ValueError, match="not valid JSON"):
            test_case_tools.do_save_test_cases("op", "not json {")

    def test_non_list_raises(self) -> None:
        from mcp_server.tools import test_case_tools

        with pytest.raises(ValueError, match="list"):
            test_case_tools.do_save_test_cases("op", json.dumps({"a": 1}))


class TestDoGetTestCases:
    def test_returns_latest(self, tmp_path: Path) -> None:
        from mcp_server.tools import test_case_tools

        test_case_tools.ensure_test_cases_schema()
        _patched_save("aclnnZeta", json.dumps([{"id": 0}]), tmp_path)
        result = test_case_tools.do_get_test_cases("aclnnZeta")
        assert result is not None
        assert result["operator_name"] == "aclnnZeta"
        assert result["cases"] == [{"id": 0}]

    def test_missing_returns_none(self) -> None:
        from mcp_server.tools import test_case_tools

        result = test_case_tools.do_get_test_cases("does_not_exist_xyz_999")
        assert result is None


class TestDoListTestCaseOperators:
    def test_lists_after_save(self, tmp_path: Path) -> None:
        from mcp_server.tools import test_case_tools

        test_case_tools.ensure_test_cases_schema()
        _patched_save("aclnnAlpha", json.dumps([]), tmp_path)
        _patched_save("aclnnBeta", json.dumps([{"id": 0}, {"id": 1}]), tmp_path)
        rows = test_case_tools.do_list_test_case_operators()
        names = {r["operator_name"] for r in rows}
        assert "aclnnAlpha" in names
        assert "aclnnBeta" in names
        # The most recently saved should be Beta or Alpha depending on insertion order;
        # we just assert both are listed.
