"""Tests for validate_results node — attribute checklist + coverage report."""

from agent.nodes.llm_description_extract.validate_results import (
    _build_coverage_report,
    _context_has_attribute,
    attribute_checklist_verify,
)


# ---------------------------------------------------------------------------
# _context_has_attribute
# ---------------------------------------------------------------------------


class TestContextHasAttribute:
    def test_direction_present(self):
        assert _context_has_attribute("This is the input tensor", "direction") is True

    def test_direction_absent(self):
        assert _context_has_attribute("A plain sentence", "direction") is False

    def test_dtype_present(self):
        assert _context_has_attribute("float32 tensor", "dtype") is True

    def test_dtype_chinese(self):
        assert _context_has_attribute("数据类型为float", "dtype") is True

    def test_shape_present(self):
        assert _context_has_attribute("shape is [N,C,H,W]", "shape") is True

    def test_optional_present(self):
        assert _context_has_attribute("This parameter is optional", "optional") is True

    def test_unknown_attr(self):
        assert _context_has_attribute("anything", "nonexistent_attr") is False


# ---------------------------------------------------------------------------
# attribute_checklist_verify
# ---------------------------------------------------------------------------


class TestAttributeChecklistVerify:
    def test_all_attributes_present(self):
        result = {
            "param_name": "x",
            "param_type": "aclTensor",
            "direction": "input",
            "llm_description": "float32 input tensor with shape [N,C], required, supports discontinuous",
            "is_support_discontinuous": '{"value": true}',
        }
        context = "input float shape required discontinuous"
        report = attribute_checklist_verify(result, context)
        assert report["param_name"] == "x"
        assert report["missing_attrs"] == []
        assert report["desc_too_short"] is False

    def test_missing_direction(self):
        result = {
            "param_name": "alpha",
            "param_type": "float32",
            "direction": "",  # missing
            "llm_description": "scaling factor",
            "is_support_discontinuous": '{"value": "N/A"}',
        }
        # Context contains direction keyword but result doesn't have it
        context = "input scaling factor for the computation"
        report = attribute_checklist_verify(result, context)
        assert "direction" in report["missing_attrs"]

    def test_no_false_positive_when_context_lacks_attr(self):
        """Should NOT flag direction as missing if context also lacks it."""
        result = {
            "param_name": "workspace",
            "param_type": "void *",
            "direction": "",
            "llm_description": "workspace memory buffer",
            "is_support_discontinuous": '{"value": "N/A"}',
        }
        # Context has no direction keywords
        context = "workspace memory for internal computation"
        report = attribute_checklist_verify(result, context)
        assert "direction" not in report["missing_attrs"]

    def test_short_description_flagged(self):
        result = {
            "param_name": "x",
            "param_type": "int32",
            "direction": "input",
            "llm_description": "short",  # < 30 chars
            "is_support_discontinuous": '{"value": "N/A"}',
        }
        context = "some context"
        report = attribute_checklist_verify(result, context)
        assert report["desc_too_short"] is True

    def test_non_tensor_skips_discontinuous_check(self):
        """Non-tensor types should not be checked for discontinuous support."""
        result = {
            "param_name": "alpha",
            "param_type": "float32",
            "direction": "input",
            "llm_description": "float scaling factor",
            "is_support_discontinuous": None,
        }
        context = "input float"
        report = attribute_checklist_verify(result, context)
        # discontinuous should NOT be in missing_attrs since param_type isn't tensor
        assert "discontinuous" not in report["missing_attrs"]


# ---------------------------------------------------------------------------
# _build_coverage_report
# ---------------------------------------------------------------------------


class TestBuildCoverageReport:
    def test_full_coverage(self):
        params = [
            {"param_name": "x"},
            {"param_name": "y"},
        ]
        results = [
            {"llm_description": "long description for x" + "x" * 30, "_validation": {"missing_attrs": []}},
            {"llm_description": "long description for y" + "y" * 30, "_validation": {"missing_attrs": []}},
        ]
        report = _build_coverage_report(results, params)
        assert report["total_params"] == 2
        assert report["extracted"] == 2
        assert report["not_extracted"] == 0
        assert report["coverage_rate"] == "100.0%"

    def test_partial_coverage(self):
        params = [
            {"param_name": "x"},
            {"param_name": "y"},
            {"param_name": "z"},
        ]
        results = [
            {"llm_description": "x" * 50, "_validation": {"missing_attrs": []}},
        ]
        report = _build_coverage_report(results, params)
        assert report["total_params"] == 3
        assert report["extracted"] == 1
        assert report["not_extracted"] == 2
        assert "33.3%" in report["coverage_rate"]

    def test_short_descriptions_counted(self):
        params = [{"param_name": "x"}, {"param_name": "y"}]
        results = [
            {"llm_description": "ok" + "x" * 50, "_validation": {"missing_attrs": []}},
            {"llm_description": "short", "_validation": {"missing_attrs": []}},
        ]
        report = _build_coverage_report(results, params)
        assert report["short_descriptions"] == 1

    def test_missing_attrs_counted(self):
        params = [{"param_name": "x"}]
        results = [
            {"llm_description": "x" * 50, "_validation": {"missing_attrs": ["shape"]}},
        ]
        report = _build_coverage_report(results, params)
        assert report["with_missing_attrs"] == 1

    def test_empty_params(self):
        report = _build_coverage_report([], [])
        assert report["total_params"] == 0
        assert report["coverage_rate"] == "N/A"
