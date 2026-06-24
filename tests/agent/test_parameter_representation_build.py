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
    _build_tensor_representations,
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
    assert (
        "alltoAllOutOptional.shape[0] == BS/rankSize.range_value"
        in exprs
    )
    assert (
        "alltoAllOutOptional.shape[1] == H*rankSize.range_value"
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
