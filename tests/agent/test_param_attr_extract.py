"""Tests for the param_attr_extract node."""

from __future__ import annotations

import json

from agent.nodes.param_attr_extract import _extract_attrs


class TestExtractDiscontinuous:
    def test_non_tensor_type_returns_na(self):
        param = {
            "param_type": "int64_t",
            "description": "| 属性 | 值 |\n|------|-----|\n| 非连续Tensor | √ |",
            "function_name": "aclnnFoo",
            "param_name": "offset",
        }
        result = _extract_attrs(param)
        assert result is not None
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] == "N/A"
        assert val["src_text"] == ""

    def test_tensor_type_with_checkmark(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 参数名 | x |\n"
                "| 非连续Tensor | √ |\n"
                "| 其他 | （无） |"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] is True
        assert val["src_text"] == "| 非连续Tensor | √ |"

    def test_tensor_type_with_checkmark(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 参数名 | x |\n"
                "| 非连续Tensor | √ |\n"
                "| 其他 | （无） |"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] is True
        assert val["src_text"] == "| 非连续Tensor | √ |"

    def test_tensor_type_with_support_text(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 非连续Tensor | 支持 |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] is True

    def test_tensor_type_with_cross(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 非连续Tensor | × |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] is False
        assert val["src_text"] == ""

    def test_tensor_type_no_row(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 参数名 | x |\n"
                "| 数据类型 | FLOAT32 |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] is False
        assert val["src_text"] == ""

    def test_empty_description_returns_none(self):
        param = {
            "param_type": "aclTensor *",
            "description": "",
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is None

    def test_tensor_case_insensitive(self):
        param = {
            "param_type": "acltensor *",
            "description": "| 属性 | 值 |\n|------|-----|\n| 非连续Tensor | √ |",
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] is True

    def test_tensor_type_with_other_text(self):
        param = {
            "param_type": "aclTensor *",
            "description": "| 属性 | 值 |\n|------|-----|\n| 非连续Tensor | 不支持 |",
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] is False
        assert val["src_text"] == ""

    def test_preserves_function_and_param_name(self):
        param = {
            "param_type": "int",
            "description": "| 属性 | 值 |\n|------|-----|",
            "function_name": "aclnnBar",
            "param_name": "scale",
        }
        result = _extract_attrs(param)
        assert result is not None
        assert result["function_name"] == "aclnnBar"
        assert result["param_name"] == "scale"


class TestExtractParamDesc:
    def test_extracts_description_row(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 参数名 | x |\n"
                "| 描述 | 输入Tensor，表示待处理的数据 |\n"
                "| 数据类型 | FLOAT32 |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        assert result["param_desc"] == "输入Tensor，表示待处理的数据"

    def test_empty_when_none_value(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 参数名 | x |\n"
                "| 描述 | （无） |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        assert result["param_desc"] == ""

    def test_empty_when_no_description_row(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 参数名 | x |\n"
                "| 数据类型 | FLOAT32 |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        assert result["param_desc"] == ""

    def test_empty_when_wu_value(self):
        param = {
            "param_type": "int64_t",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 描述 | 无 |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "offset",
        }
        result = _extract_attrs(param)
        assert result is not None
        assert result["param_desc"] == ""


class TestCombinedExtraction:
    def test_both_fields_extracted(self):
        param = {
            "param_type": "aclTensor *",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 参数名 | x |\n"
                "| 描述 | 输入Tensor |\n"
                "| 数据类型 | FLOAT32 |\n"
                "| 非连续Tensor | √ |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "x",
        }
        result = _extract_attrs(param)
        assert result is not None
        assert result["param_desc"] == "输入Tensor"
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] is True

    def test_non_tensor_with_desc(self):
        param = {
            "param_type": "int64_t",
            "description": (
                "| 属性 | 值 |\n"
                "|------|-----|\n"
                "| 参数名 | offset |\n"
                "| 描述 | 偏移量，单位为字节 |\n"
                "| 非连续Tensor | √ |\n"
            ),
            "function_name": "aclnnFoo",
            "param_name": "offset",
        }
        result = _extract_attrs(param)
        assert result is not None
        assert result["param_desc"] == "偏移量，单位为字节"
        val = json.loads(result["is_support_discontinuous"])
        assert val["value"] == "N/A"
