"""Tests for verify_and_enhance node."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent.nodes.llm_description_extract.verify_and_enhance import (
    _build_audit_record,
    _needs_verification,
    _verify_one,
    verify_and_enhance_node,
)


# ---------------------------------------------------------------------------
# _needs_verification
# ---------------------------------------------------------------------------

class TestNeedsVerification:
    def test_missing_attrs_triggers(self):
        result = {"_validation": {"missing_attrs": ["shape"]}}
        assert _needs_verification(result) is True

    def test_short_desc_triggers(self):
        result = {"_validation": {"desc_too_short": True, "missing_attrs": []}}
        assert _needs_verification(result) is True

    def test_clean_result_skips(self):
        result = {"_validation": {"missing_attrs": [], "desc_too_short": False}}
        assert _needs_verification(result) is False

    def test_no_validation_skips(self):
        result = {}
        assert _needs_verification(result) is False

    def test_both_triggers(self):
        result = {"_validation": {"missing_attrs": ["dtype"], "desc_too_short": True}}
        assert _needs_verification(result) is True


# ---------------------------------------------------------------------------
# _build_audit_record
# ---------------------------------------------------------------------------

class TestBuildAuditRecord:
    def test_not_triggered_audit(self):
        result = {"llm_description": "x is input tensor", "src_content": "src"}
        validation = {"missing_attrs": [], "desc_too_short": False}
        audit = _build_audit_record(result, validation, None, False)
        assert audit["extraction"]["original_description"] == "x is input tensor"
        assert audit["extraction"]["char_count"] == 17
        assert audit["validation"]["triggered"] is False
        assert audit["verification"]["triggered"] is False

    def test_triggered_but_not_enhanced(self):
        result = {
            "llm_description": "short",
            "_original_description": "short",
            "_original_src_content": "",
            "_validation": {"missing_attrs": ["shape"], "desc_too_short": True},
        }
        validation = {"missing_attrs": ["shape"], "desc_too_short": True}
        verify_resp = {"has_missing_info": False, "found_attrs": [], "enhanced_description": ""}
        audit = _build_audit_record(result, validation, verify_resp, False)
        assert audit["validation"]["triggered"] is True
        assert audit["verification"]["triggered"] is True
        assert audit["verification"]["has_missing_info"] is False
        assert audit["verification"]["enhanced"] is False

    def test_enhanced_audit(self):
        result = {
            "llm_description": "enhanced description with more info",
            "_original_description": "short",
            "_original_src_content": "orig src",
            "_validation": {"missing_attrs": ["shape"], "desc_too_short": True},
        }
        validation = {"missing_attrs": ["shape"], "desc_too_short": True}
        verify_resp = {
            "has_missing_info": True,
            "found_attrs": ["shape"],
            "enhanced_description": "enhanced description with more info",
        }
        audit = _build_audit_record(result, validation, verify_resp, True)
        assert audit["extraction"]["original_description"] == "short"
        assert audit["extraction"]["original_src_content"] == "orig src"
        assert audit["verification"]["enhanced"] is True
        assert audit["verification"]["found_attrs"] == ["shape"]
        assert audit["verification"]["enhanced_char_count"] == 35

    def test_uses_original_snapshot_fields(self):
        """When _original_description is set, it takes priority over llm_description."""
        result = {
            "llm_description": "modified",
            "_original_description": "original",
            "_original_src_content": "orig src",
        }
        audit = _build_audit_record(result, {}, None, False)
        assert audit["extraction"]["original_description"] == "original"
        assert audit["extraction"]["char_count"] == 8


# ---------------------------------------------------------------------------
# _verify_one
# ---------------------------------------------------------------------------

class TestVerifyOne:
    @pytest.mark.asyncio
    async def test_skips_when_not_triggered(self):
        """No _validation issues -> skip LLM call, build audit."""
        result = {
            "param_name": "x",
            "llm_description": "x is a complete description with enough chars",
            "_validation": {"missing_attrs": [], "desc_too_short": False},
            "_context": "some context",
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()
        updated = await _verify_one(mock_llm, result)
        mock_llm.ainvoke.assert_not_called()
        assert "description_audit" in updated
        assert updated["description_audit"]["verification"]["triggered"] is False

    @pytest.mark.asyncio
    async def test_skips_when_no_context(self):
        """Empty _context -> skip LLM call."""
        result = {
            "param_name": "x",
            "llm_description": "short",
            "_validation": {"missing_attrs": ["shape"], "desc_too_short": True},
            "_context": "",
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()
        updated = await _verify_one(mock_llm, result)
        mock_llm.ainvoke.assert_not_called()
        assert updated["llm_description"] == "short"

    @pytest.mark.asyncio
    async def test_enhances_when_doc_has_info(self):
        """LLM says doc has missing info -> replace description."""
        result = {
            "param_name": "x",
            "llm_description": "x is input tensor",
            "src_content": "original src",
            "_validation": {"missing_attrs": ["shape"], "desc_too_short": False},
            "_context": "| x | aclTensor | shape [N,C,H,W] |",
        }
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"has_missing_info": true, "found_attrs": ["shape"], "enhanced_description": "x is input tensor with shape [N,C,H,W]", "enhanced_src_content": "shape [N,C,H,W]"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        updated = await _verify_one(mock_llm, result)
        assert updated["llm_description"] == "x is input tensor with shape [N,C,H,W]"
        assert updated["_enhanced"] is True
        assert "shape [N,C,H,W]" in updated["src_content"]
        assert updated["description_audit"]["verification"]["enhanced"] is True

    @pytest.mark.asyncio
    async def test_keeps_original_when_doc_lacks_info(self):
        """LLM says doc does NOT have missing info -> keep original."""
        result = {
            "param_name": "x",
            "llm_description": "x is input tensor",
            "_validation": {"missing_attrs": ["discontinuous"], "desc_too_short": False},
            "_context": "| x | aclTensor | input |",
        }
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"has_missing_info": false, "found_attrs": [], "enhanced_description": ""}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        updated = await _verify_one(mock_llm, result)
        assert updated["llm_description"] == "x is input tensor"
        assert not updated.get("_enhanced")

    @pytest.mark.asyncio
    async def test_keeps_original_when_enhanced_shorter(self):
        """Enhanced desc shorter than original -> keep original."""
        result = {
            "param_name": "x",
            "llm_description": "x is a fairly complete input tensor description",
            "_validation": {"missing_attrs": ["shape"], "desc_too_short": False},
            "_context": "some context",
        }
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"has_missing_info": true, "found_attrs": ["shape"], "enhanced_description": "short"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        updated = await _verify_one(mock_llm, result)
        assert "fairly complete" in updated["llm_description"]
        assert not updated.get("_enhanced")

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        """LLM call fails -> keep original, still build audit."""
        result = {
            "param_name": "x",
            "llm_description": "x is input",
            "_validation": {"missing_attrs": ["shape"], "desc_too_short": False},
            "_context": "some context",
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        updated = await _verify_one(mock_llm, result)
        assert updated["llm_description"] == "x is input"
        assert "description_audit" in updated


# ---------------------------------------------------------------------------
# verify_and_enhance_node
# ---------------------------------------------------------------------------

class TestVerifyAndEnhanceNode:
    @pytest.mark.asyncio
    async def test_no_results_returns_clean(self):
        state = {"ws_results": [], "exe_results": []}
        result = await verify_and_enhance_node(state)
        assert result["enhance_count"] == 0
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_all_clean_no_llm_calls(self):
        """All results pass validation -> no LLM calls."""
        ws_results = [
            {
                "param_name": "x",
                "llm_description": "complete description",
                "_validation": {"missing_attrs": [], "desc_too_short": False},
                "_context": "ctx",
            }
        ]
        state = {"ws_results": ws_results, "exe_results": []}
        result = await verify_and_enhance_node(state)
        assert result["enhance_count"] == 0
        # Audit should still be built
        assert "description_audit" in ws_results[0]
        assert ws_results[0]["description_audit"]["validation"]["triggered"] is False

    @pytest.mark.asyncio
    async def test_builds_audit_for_all_results(self):
        """Audit records are built for both triggered and non-triggered results."""
        ws_results = [
            {
                "param_name": "x",
                "llm_description": "complete",
                "_validation": {"missing_attrs": [], "desc_too_short": False},
                "_context": "ctx",
            },
            {
                "param_name": "y",
                "llm_description": "short",
                "_validation": {"missing_attrs": ["shape"], "desc_too_short": True},
                "_context": "",
            },
        ]
        state = {"ws_results": ws_results, "exe_results": []}
        result = await verify_and_enhance_node(state)
        # Both should have audit records
        assert "description_audit" in ws_results[0]
        assert "description_audit" in ws_results[1]
        # y was triggered but had no context, so not enhanced
        assert ws_results[1]["description_audit"]["validation"]["triggered"] is True


# Write to target file
