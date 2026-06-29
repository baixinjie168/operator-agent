"""Tests for quantization_type implicit parameter extraction."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure agent package is importable
ROOT = Path(__file__).resolve().parents[2]
for pkg in ("packages/shared/src", "packages/mcp-server/src", "packages/agent/src"):
    p = str(ROOT / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.nodes.param_relation_extract.implicit_param_extract import (  # noqa: E402
    _QUANTIZATION_CANDIDATES,
    _apply_agent_actions,
    _build_quantization_type_mapping,
    _build_sig_param_types,
    _extract_quantization_modes,
    _identify_tensor_params,
)

# ---------------------------------------------------------------------------
# _extract_quantization_modes
# ---------------------------------------------------------------------------


class TestExtractQuantizationModes:
    def test_full_hit_all_four(self):
        text = "支持 per-channel、per-group、per-tensor、per-token 四种模式"
        modes, _ = _extract_quantization_modes(text)
        assert modes == ["per-channel", "per-group", "per-tensor", "per-token"]

    def test_partial_hit_preserves_canonical_order(self):
        # Document mentions per-tensor before per-channel, but result keeps
        # the canonical _QUANTIZATION_CANDIDATES order.
        text = "支持per-tensor，per-channel"
        modes, _ = _extract_quantization_modes(text)
        assert modes == ["per-channel", "per-tensor"]

    def test_single_hit_per_group(self):
        text = "per-group下输入为二维向量"
        modes, _ = _extract_quantization_modes(text)
        assert modes == ["per-group"]

    def test_no_hit_returns_empty(self):
        text = "这是一个没有量化粒度关键词的普通算子说明文档。"
        modes, _ = _extract_quantization_modes(text)
        assert modes == []

    def test_empty_text_returns_empty(self):
        modes, _ = _extract_quantization_modes("")
        assert modes == []

    def test_deduplication_repeated(self):
        text = "per-channel ... per-channel ... per-channel"
        modes, _ = _extract_quantization_modes(text)
        assert modes == ["per-channel"]

    def test_word_boundary_no_false_positive_percent(self):
        """'percent-channel' must not match 'per-channel'."""
        text = "使用 percent-channel 编码"
        modes, _ = _extract_quantization_modes(text)
        assert modes == []

    def test_word_boundary_no_false_positive_substring(self):
        """'super-channel' must not match 'per-channel'."""
        text = "super-channel 模式"
        modes, _ = _extract_quantization_modes(text)
        assert modes == []

    def test_source_citation_returned(self):
        text = "前缀文字 per-tensor 后缀文字"
        _, citation = _extract_quantization_modes(text)
        assert "per-tensor" in citation

    def test_candidates_constant_has_four_values(self):
        assert _QUANTIZATION_CANDIDATES == (
            "per-channel", "per-group", "per-tensor", "per-token",
        )


# ---------------------------------------------------------------------------
# _build_quantization_type_mapping
# ---------------------------------------------------------------------------


class TestBuildQuantizationTypeMapping:
    def test_structure_with_hits(self):
        text = "支持 per-channel、per-tensor 两种"
        m = _build_quantization_type_mapping(text)
        assert m["var_name"] == "quantization_type"
        assert m["is_quantization_type"] is True
        assert m["param_type"] == "char"
        assert m["allowed_range_type"] == "enum"
        assert m["allowed_range_value"] == ["per-channel", "per-tensor"]
        assert m["is_constant"] is False
        assert m["is_external_constant"] is False
        assert m["is_compound"] is False
        assert m["tensor_param"] is None
        assert m["dim_index"] is None
        assert "per-channel" in m["source_citation"]

    def test_empty_allowed_range_when_no_hit(self):
        """No quantization terms → empty allowed_range_value, but still present."""
        m = _build_quantization_type_mapping("普通算子说明，无量化词")
        assert m["var_name"] == "quantization_type"
        assert m["allowed_range_value"] == []
        assert m["allowed_range_type"] == "enum"
        assert m["param_type"] == "char"
        assert m["source_citation"] == ""

    def test_full_four_modes(self):
        text = "per-channel per-group per-tensor per-token"
        m = _build_quantization_type_mapping(text)
        assert m["allowed_range_value"] == [
            "per-channel", "per-group", "per-tensor", "per-token",
        ]

    def test_empty_text_still_returns_mapping(self):
        m = _build_quantization_type_mapping("")
        assert m["var_name"] == "quantization_type"
        assert m["allowed_range_value"] == []


# ---------------------------------------------------------------------------
# Item 1: tensor_param source-defense validation
# ---------------------------------------------------------------------------


class TestBuildSigParamTypes:
    """_build_sig_param_types: build {name: type} from normalized signatures."""

    def test_extracts_name_and_type(self):
        sigs = [{
            "function_name": "GetWorkspaceSize",
            "parameters": [
                {"name": "x1", "type": "aclTensor"},
                {"name": "transposeX2", "type": "bool"},
                {"name": "alltoAllAxesOptional", "type": "aclIntArray"},
            ],
        }]
        types = _build_sig_param_types(sigs)
        assert types["x1"] == "aclTensor"
        assert types["transposeX2"] == "bool"
        assert types["alltoAllAxesOptional"] == "aclIntArray"
        # snake_case alias also present
        assert types["transpose_x2"] == "bool"

    def test_empty_signatures(self):
        assert _build_sig_param_types([]) == {}


class TestIdentifyTensorParamsSignaturePriority:
    """T1-1/T1-2: signature types are authoritative over HTML heuristic."""

    def test_t1_1_signature_authoritative_excludes_bool(self):
        """transposeX2(bool) HTML cell mentions 'shape' but signature says bool."""
        html = (
            "<table><tr>"
            "<td>x1（aclTensor*）</td>"
            "<td>shape (BS, H)</td>"
            "</tr><tr>"
            "<td>transposeX2（bool）</td>"
            "<td>shape说明: 与x1的shape关系</td>"
            "</tr></table>"
        )
        sig_params = {"x1", "transposeX2", "transpose_x2"}
        sig_types = {"x1": "aclTensor", "transposeX2": "bool",
                     "transpose_x2": "bool"}
        result = _identify_tensor_params(html, sig_params, sig_types)
        assert "x1" in result
        # transposeX2 is bool in signature → excluded despite HTML "shape"
        assert "transposeX2" not in result

    def test_t1_2_no_signature_falls_back_to_html(self):
        """When sig_param_types is None, HTML heuristic is used."""
        html = (
            "<table><tr>"
            "<td>x1</td>"
            "<td>FLOAT16 shape (BS, H)</td>"
            "</tr></table>"
        )
        sig_params = {"x1"}
        result = _identify_tensor_params(html, sig_params, None)
        assert "x1" in result

    def test_untyped_param_rescued_by_html(self):
        """A param absent from sig_types can still be found via HTML."""
        html = (
            "<table><tr>"
            "<td>mystery</td>"
            "<td>FLOAT16 shape (N)</td>"
            "</tr></table>"
        )
        sig_params = {"mystery"}
        sig_types = {"x1": "aclTensor"}  # mystery not in types
        result = _identify_tensor_params(html, sig_params, sig_types)
        assert "mystery" in result


class TestApplyAgentActionsTensorGuard:
    """T1-3/T1-4: _apply_agent_actions strips non-Tensor tensor_param."""

    @staticmethod
    def _candidate(cid, var, tensor, dim):
        return {
            "candidate_id": cid,
            "var_name": var,
            "tensor_param": tensor,
            "dim_index": dim,
            "slot_index": dim,
            "slot_expr": "N",
            "is_compound": False,
            "compound_expr": None,
            "shape_text": "(N)",
        }

    def test_t1_3_agent_mislabels_bool_strips_shape_mapping(self):
        """Agent confirm with non-Tensor tensor_param → shape stripped."""
        candidates = [self._candidate("cand_001", "N", "transposeX2", 0)]
        parsed = {
            "actions": [{
                "candidate_id": "cand_001",
                "action": "confirm",
                "classification": "dimension_variable",
                "var_name": "N",
                "tensor_param": "transposeX2",
                "dim_index": 0,
                "reason": "test",
            }],
            "additions": [],
        }
        mappings = _apply_agent_actions(
            candidates, parsed, tensor_params={"x1", "x2"},
        )
        assert len(mappings) == 1
        m = mappings[0]
        # Variable retained as dimension_variable...
        assert m["var_name"] == "N"
        # ...but shape mapping stripped
        assert m["tensor_param"] is None
        assert m["dim_index"] is None
        assert m["shape_text"] is None
        assert m["slot_expr"] is None

    def test_t1_4_addition_guard_strips_non_tensor(self):
        """Agent addition with non-Tensor tensor_param → nulled."""
        parsed = {
            "actions": [],
            "additions": [{
                "var_name": "BS",
                "classification": "dimension_variable",
                "tensor_param": "alltoAllAxesOptional",
                "dim_index": 0,
                "reason": "test",
            }],
        }
        mappings = _apply_agent_actions(
            [], parsed, tensor_params={"x1"},
        )
        assert len(mappings) == 1
        m = mappings[0]
        assert m["var_name"] == "BS"
        assert m["tensor_param"] is None
        assert m["dim_index"] is None

    def test_no_guard_when_tensor_params_none(self):
        """When tensor_params is None, no guarding (degradation)."""
        candidates = [self._candidate("cand_001", "N", "transposeX2", 0)]
        parsed = {
            "actions": [{
                "candidate_id": "cand_001",
                "action": "confirm",
                "classification": "dimension_variable",
                "var_name": "N",
                "reason": "test",
            }],
            "additions": [],
        }
        mappings = _apply_agent_actions(candidates, parsed, tensor_params=None)
        assert len(mappings) == 1
        # No stripping happened
        assert mappings[0]["tensor_param"] == "transposeX2"

    def test_external_constant_exempt_from_guard(self):
        """External constants have tensor_param=None already; not affected."""
        parsed = {
            "actions": [],
            "additions": [{
                "var_name": "rankSize",
                "classification": "external_constant",
                "tensor_param": None,
                "dim_index": None,
                "reason": "test",
            }],
        }
        mappings = _apply_agent_actions(
            [], parsed, tensor_params={"x1"},
        )
        assert len(mappings) == 1
        assert mappings[0]["is_external_constant"] is True
