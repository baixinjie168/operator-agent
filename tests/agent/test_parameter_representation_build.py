"""Tests for the deterministic parameter_representation builder."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure agent package is importable
ROOT = Path(__file__).resolve().parents[2]
for pkg in ("packages/shared/src", "packages/mcp-server/src", "packages/agent/src"):
    p = str(ROOT / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.nodes.param_relation_extract.parameter_representation_build import (  # noqa: E402
    _build_platform_constant_representations,
    _build_tensor_param_names,
    _build_tensor_representations,
    _detect_shape_guard,
    _slot_expr_to_python,
)

# ---------------------------------------------------------------------------
# _slot_expr_to_python
# ---------------------------------------------------------------------------


def test_slot_expr_single_var():
    py, vars_ = _slot_expr_to_python("BS", {})
    assert py == "BS"
    assert vars_ == ["BS"]


def test_slot_expr_compound_multiply():
    py, vars_ = _slot_expr_to_python("H*rankSize", {})
    assert py == "H*rankSize"
    assert vars_ == ["H", "rankSize"]


def test_slot_expr_compound_divide():
    py, vars_ = _slot_expr_to_python("BS/rankSize", {})
    assert py == "BS/rankSize"
    assert vars_ == ["BS", "rankSize"]


def test_slot_expr_constant_substitution():
    py, vars_ = _slot_expr_to_python("k1*k0", {"k0": 16})
    assert py == "k1*16"
    assert vars_ == ["k1"]


def test_slot_expr_excludes_keywords():
    py, vars_ = _slot_expr_to_python("shape", {})
    assert py == "shape"
    assert vars_ == []


def test_slot_expr_external_const_keeps_range_value():
    """External constants keep .range_value; dimension variables do not."""
    ext = {"rankSize"}
    py, vars_ = _slot_expr_to_python("H*rankSize", {}, ext)
    assert py == "H*rankSize.range_value"
    assert vars_ == ["H", "rankSize"]

    py, vars_ = _slot_expr_to_python("BS/rankSize", {}, ext)
    assert py == "BS/rankSize.range_value"
    assert vars_ == ["BS", "rankSize"]


# ---------------------------------------------------------------------------
# _build_tensor_representations — aclnnAlltoAllMatmul scenario
# ---------------------------------------------------------------------------


def _mk(var, tensor, dim, slot_expr, is_compound, shape_text):
    return {
        "var_name": var,
        "tensor_param": tensor,
        "dim_index": dim,
        "slot_index": dim,
        "slot_expr": slot_expr,
        "is_compound": is_compound,
        "shape_text": shape_text,
        "is_constant": False,
        "constant_value": None,
        "is_external_constant": False,
    }


ALLTOALL_MAPPINGS = [
    # x1: (BS, H)
    _mk("BS", "x1", 0, "BS", False, "(BS, H)"),
    _mk("H", "x1", 1, "H", False, "(BS, H)"),
    # x2: (H*rankSize, N)
    _mk("H", "x2", 0, "H*rankSize", True, "(H*rankSize, N)"),
    _mk("rankSize", "x2", 0, "H*rankSize", True, "(H*rankSize, N)"),
    _mk("N", "x2", 1, "N", False, "(H*rankSize, N)"),
    # biasOptional: (N)
    _mk("N", "biasOptional", 0, "N", False, "(N)"),
    # output: (BS/rankSize, N)
    _mk("BS", "output", 0, "BS/rankSize", True, "(BS/rankSize, N)"),
    _mk("rankSize", "output", 0, "BS/rankSize", True, "(BS/rankSize, N)"),
    _mk("N", "output", 1, "N", False, "(BS/rankSize, N)"),
    # alltoAllOutOptional: (BS/rankSize, H*rankSize)
    _mk("BS", "alltoAllOutOptional", 0, "BS/rankSize", True,
        "(BS/rankSize, H*rankSize)"),
    _mk("H", "alltoAllOutOptional", 1, "H*rankSize", True,
        "(BS/rankSize, H*rankSize)"),
    # rankSize as external constant (should be skipped)
    {
        "var_name": "rankSize",
        "tensor_param": None,
        "dim_index": None,
        "slot_index": None,
        "slot_expr": None,
        "is_compound": False,
        "shape_text": None,
        "is_constant": False,
        "constant_value": None,
        "is_external_constant": True,
    },
]


def test_alltoall_tensor_representations():
    reps = _build_tensor_representations(ALLTOALL_MAPPINGS)

    # Expected 9 unique (tensor, dim) slots with at least one var:
    # x1[0], x1[1], x2[0], x2[1], biasOptional[0],
    # output[0], output[1], alltoAllOutOptional[0], alltoAllOutOptional[1]
    assert len(reps) == 9, "expected 9 reps, got " + str(len(reps))

    exprs = [r["expr"] for r in reps]
    assert "x1.shape[0] == BS" in exprs
    assert "output.shape[0] == BS/rankSize.range_value" in exprs
    # Item 5: optional params (name contains "Optional") get a guard.
    assert (
        "(alltoAllOutOptional.shape[0] == BS/rankSize.range_value) "
        "if alltoAllOutOptional is not None else True"
        in exprs
    )
    assert (
        "(alltoAllOutOptional.shape[1] == H*rankSize.range_value) "
        "if alltoAllOutOptional is not None else True"
        in exprs
    )
    assert (
        "(biasOptional.shape[0] == N) "
        "if biasOptional is not None else True"
        in exprs
    )

    for r in reps:
        assert r["expr_type"] == "parameter_representation"
        assert r["relation_params"][0]  # tensor name first


def test_duplicate_slot_dedup():
    """A (tensor, dim) slot appearing twice must yield one rep."""
    mappings = [
        _mk("BS", "x1", 0, "BS", False, "(BS, H)"),
        _mk("H", "x1", 0, "BS", False, "(BS, H)"),
    ]
    reps = _build_tensor_representations(mappings)
    assert len(reps) == 1


def test_empty_mappings_returns_empty():
    assert _build_tensor_representations([]) == []


def test_only_constants_skipped():
    """Pure-constant mappings (k0=16) must produce no reps."""
    m = _mk("k0", "x", 0, "k0", False, "(k0)")
    m["is_constant"] = True
    m["constant_value"] = 16
    assert _build_tensor_representations([m]) == []


# ---------------------------------------------------------------------------
# _build_platform_constant_representations
# ---------------------------------------------------------------------------


PLATFORM_CONSTANTS = [
    {
        "const_name": "rankSize",
        "description": "NPU card count",
        "platform_values": [
            {
                "platform": "Atlas A2 训练系列产品/Atlas A2 推理系列产品",
                "values": [2, 4, 8],
                "source_citation": "支持2、4、8卡",
            },
            {
                "platform": "Atlas 350 加速卡",
                "values": [2, 4, 8, 16],
                "source_citation": "支持2、4、8、16卡",
            },
        ],
    },
]


def test_platform_constant_representations():
    by_platform = _build_platform_constant_representations(PLATFORM_CONSTANTS)
    a2_key = "Atlas A2 训练系列产品/Atlas A2 推理系列产品"
    assert a2_key in by_platform
    assert "Atlas 350 加速卡" in by_platform

    a2 = by_platform[a2_key]
    assert len(a2) == 1
    assert a2[0]["expr"] == "rankSize.range_value in [2, 4, 8]"
    assert a2[0]["expr_type"] == "parameter_representation"
    assert a2[0]["relation_params"] == ["rankSize"]

    a350 = by_platform["Atlas 350 加速卡"]
    assert a350[0]["expr"] == "rankSize.range_value in [2, 4, 8, 16]"


def test_empty_platform_constants():
    assert _build_platform_constant_representations([]) == {}


# ---------------------------------------------------------------------------
# Item 0: Tensor type filtering (downstream defense)
# ---------------------------------------------------------------------------


class TestBuildTensorParamNames:
    """_build_tensor_param_names: classify aclTensor* params from type map."""

    def test_returns_only_tensor_params(self):
        param_types = {
            "x1": "aclTensor",
            "transposeX2": "bool",
            "alltoAllAxesOptional": "aclIntArray",
            "weight": "aclTensorList",
        }
        names = _build_tensor_param_names(param_types)
        assert names == {"x1", "weight"}

    def test_empty_map_returns_empty(self):
        assert _build_tensor_param_names({}) == set()


class TestTensorTypeFiltering:
    """Item 0: non-Tensor tensor_param values are skipped (T0-1..T0-4)."""

    def test_t0_1_normal_tensor_generates_constraint(self):
        """T0-1: a real aclTensor param produces a shape constraint."""
        mappings = [_mk("BS", "x1", 0, "BS", False, "(BS, H)")]
        reps = _build_tensor_representations(mappings, {"x1"})
        assert len(reps) == 1
        assert reps[0]["expr"] == "x1.shape[0] == BS"

    def test_t0_2_misclassified_bool_skipped(self):
        """T0-2: transposeX2(bool) misclassified as tensor_param is skipped."""
        mappings = [_mk("N", "transposeX2", 0, "N", False, "(N)")]
        reps = _build_tensor_representations(mappings, {"x1", "x2"})
        assert reps == []  # transposeX2 not in tensor set → skipped

    def test_t0_3_misclassified_aclintarray_skipped(self):
        """T0-3: alltoAllAxesOptional(aclIntArray) is skipped."""
        mappings = [_mk("BS", "alltoAllAxesOptional", 0, "BS", False, "(BS)")]
        reps = _build_tensor_representations(mappings, {"x1"})
        assert reps == []

    def test_t0_4_degradation_none_keeps_original_behavior(self):
        """T0-4: tensor_param_names=None degrades to permissive behavior."""
        mappings = [_mk("N", "transposeX2", 0, "N", False, "(N)")]
        # None = type query failed → original permissive behavior, no filtering
        reps = _build_tensor_representations(mappings, None)
        assert len(reps) == 1
        assert reps[0]["expr"] == "transposeX2.shape[0] == N"

    def test_mixed_tensor_and_non_tensor(self):
        """A mix: only real Tensor params produce reps."""
        mappings = [
            _mk("BS", "x1", 0, "BS", False, "(BS, H)"),
            _mk("N", "transposeX2", 0, "N", False, "(N)"),
            _mk("H", "x2", 1, "H", False, "(H)"),
        ]
        reps = _build_tensor_representations(mappings, {"x1", "x2"})
        assert len(reps) == 2
        exprs = [r["expr"] for r in reps]
        assert "x1.shape[0] == BS" in exprs
        assert "x2.shape[1] == H" in exprs
        assert not any("transposeX2" in e for e in exprs)


# ---------------------------------------------------------------------------
# Item 5: _detect_shape_guard — condition guard for optional / conditional params
# ---------------------------------------------------------------------------

class TestDetectShapeGuard:
    """Item 5: condition-guard detection (T5-1..T5-6)."""

    def test_t5_1_optional_flag(self):
        """T5-1: is_optional=True → '{tensor} is not None'."""
        guard = _detect_shape_guard("scaleOptional", "", "", is_optional=True)
        assert guard == "scaleOptional is not None"

    def test_t5_2_name_contains_optional(self):
        """T5-2: name contains 'Optional' even without is_optional flag."""
        guard = _detect_shape_guard("biasOptional", "", "", is_optional=False)
        assert guard == "biasOptional is not None"

    def test_t5_3_per_channel(self):
        """T5-3: param_desc mentions 'per-channel'."""
        guard = _detect_shape_guard(
            "scale", "", "per-channel时shape为(E,K1,N1)", is_optional=False,
        )
        assert guard == "per_channel"

    def test_t5_3b_per_group(self):
        guard = _detect_shape_guard(
            "scale", "", "per-group时shape为(E,K1,N1)", is_optional=False,
        )
        assert guard == "per_group"

    def test_t5_4_has_expert(self):
        """T5-4: param_desc mentions '有专家'."""
        guard = _detect_shape_guard(
            "scale", "", "有专家时shape为(E,K1,N1)", is_optional=False,
        )
        assert guard == "expertTokens is not None"

    def test_t5_4b_no_expert(self):
        guard = _detect_shape_guard(
            "scale", "", "无专家时shape为(N1)", is_optional=False,
        )
        assert guard == "expertTokens is None"

    def test_t5_5_no_condition(self):
        """T5-5: no keyword → empty guard (original behavior)."""
        guard = _detect_shape_guard("x1", "(BS, H)", "普通参数", is_optional=False)
        assert guard == ""

    def test_t5_6_degradation_empty_maps(self):
        """T5-6: empty param_desc/shape_text + non-optional name → no guard."""
        guard = _detect_shape_guard("x1", "", "", is_optional=False)
        assert guard == ""

    def test_per_tensor_skipped(self):
        """per-tensor is the 'else' branch — skipped (covered by per-channel)."""
        guard = _detect_shape_guard(
            "scale", "", "per-tensor时shape为(N1)", is_optional=False,
        )
        assert guard == ""

    def test_optional_takes_priority_over_keyword(self):
        """Optional flag is checked first (most reliable)."""
        guard = _detect_shape_guard(
            "scaleOptional", "", "per-channel时shape为(E,K1)", is_optional=True,
        )
        assert guard == "scaleOptional is not None"


class TestBuildTensorRepresentationsGuard:
    """Item 5: _build_tensor_representations wraps expr with guard."""

    def test_optional_param_gets_guard(self):
        """Optional param name → expr wrapped with 'if ... is not None else True'."""
        mappings = [_mk("N", "scaleOptional", 0, "N", False, "(N)")]
        reps = _build_tensor_representations(mappings, {"scaleOptional"})
        assert len(reps) == 1
        assert "if scaleOptional is not None else True" in reps[0]["expr"]

    def test_per_channel_param_gets_guard(self):
        """param_desc with per-channel → expr wrapped with 'if per_channel else True'."""
        mappings = [_mk("E", "scale", 0, "E", False, "(E,K1,N1)")]
        param_descs = {"scale": "per-channel时shape为(E,K1,N1)"}
        reps = _build_tensor_representations(
            mappings, {"scale"}, param_descs=param_descs,
        )
        assert len(reps) == 1
        assert "if per_channel else True" in reps[0]["expr"]

    def test_no_guard_for_plain_param(self):
        """Plain param with no conditions → original unguarded expr."""
        mappings = [_mk("BS", "x1", 0, "BS", False, "(BS, H)")]
        reps = _build_tensor_representations(mappings, {"x1"})
        assert len(reps) == 1
        assert reps[0]["expr"] == "x1.shape[0] == BS"

    def test_degradation_no_param_descs(self):
        """When param_descs is None/empty, only name-based optional detection works."""
        mappings = [_mk("N", "scale", 0, "N", False, "(N)")]
        # No param_descs → per-channel in desc not detected; "scale" not optional
        reps = _build_tensor_representations(mappings, {"scale"})
        assert len(reps) == 1
        assert "if " not in reps[0]["expr"]

