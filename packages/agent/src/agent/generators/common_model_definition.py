# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
版权信息：华为技术有限公司，版本所有(C) 2025-2025
修改记录：2026/6/24
功能：算子约束与参数间约束共用数据模型定义

本模块从 operator_case_generator 迁移而来，定义了下游算子用例生成所需的所有
Pydantic 数据模型，包括：

- ValueWithSrcText      参数属性封装
- ParamAttributes       单个参数在某个平台下的属性集
- InterParamConstraint  参数间约束表达式
- OperatorRule          算子完整约束数据顶层结构
- InterConstraintsRuleType 参数间约束类型枚举
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# 参数属性与约束共用模型
# ─────────────────────────────────────────────────────────────────────


class ValueWithSrcText(BaseModel):
    """带 src_text 溯源信息的参数属性取值。

    大多数参数属性形如 ``{"value": [...], "src_text": "..."}``，使用
    ValueWithSrcText 可以让 Pydantic 自动将其解析为对象。
    """

    value: Optional[Union[str, int, float, bool, list, dict]] = None
    type: Optional[str] = None
    src_text: Optional[str] = None


class ParamAttributes(BaseModel):
    """单个参数在某一平台上的完整属性集合。"""

    description: Optional[Union[str, ValueWithSrcText]] = None
    type: Optional[Union[str, ValueWithSrcText]] = None
    format: Optional[Union[str, ValueWithSrcText]] = None
    is_optional: Optional[Union[bool, ValueWithSrcText]] = None
    is_support_discontinuous: Optional[Union[str, ValueWithSrcText]] = None
    is_operator_param: Optional[Union[bool, ValueWithSrcText]] = None
    dimensions: Optional[Union[list, ValueWithSrcText]] = None
    array_length: Optional[Union[str, ValueWithSrcText]] = None
    dtype: Optional[Union[list, ValueWithSrcText]] = None
    allowed_range_value: Optional[Union[list, ValueWithSrcText]] = None

    model_config = {"extra": "allow"}


# ─────────────────────────────────────────────────────────────────────
# 参数间约束
# ─────────────────────────────────────────────────────────────────────


class InterConstraintsRuleType(Enum):
    """参数间约束类型枚举。"""

    SHAPE_EQUALITY = "shape_equality"
    SHAPE_CHOICE = "shape_choice"
    SHAPE_BROADCAST = "shape_broadcast"
    SHAPE_DEPENDENCY = "shape_dependency"
    TYPE_EQUALITY = "type_equality"
    TYPE_DEPENDENCY = "type_dependency"
    VALUE_DEPENDENCY = "value_dependency"
    PRESENCE_DEPENDENCY = "presence_dependency"
    FORMAT_EQUALITY = "format_equality"


class InterParamConstraint(BaseModel):
    """参数间约束表达式。

    通过 ``relation_params`` 描述表达式涉及的参数，通过 ``expr_type`` 描述
    约束类型，``expr`` 字段保存具体约束表达式字符串。
    """

    relation_params: List[str] = Field(default_factory=list)
    expr_type: str = ""
    expr: str = ""
    description: str = ""

    model_config = {"extra": "allow"}


# ─────────────────────────────────────────────────────────────────────
# 算子完整约束数据
# ─────────────────────────────────────────────────────────────────────


class OperatorRule(BaseModel):
    """算子完整约束数据顶层结构。

    该对象对应 ``assemble_result`` 写出的 ``result.json`` 形态：

    - ``operator_name``                算子名称
    - ``inputs`` / ``outputs``         参数在每个平台上的属性集
    - ``constraints_in_parameters``    参数间约束（per-platform）
    - ``dtype_support_description``    每平台 dtype 组合
    - ``format_support_description``   每平台 format 组合
    - ``deterministic_computing``      平台确定性计算规则
    - 其余字段透传保留，便于生成阶段使用
    """

    operator_name: str = ""
    function_explanation: Any = None
    product_support: List[str] = Field(default_factory=list)
    function_signature: str = ""
    deterministic_computing: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Dict[str, ParamAttributes]] = Field(default_factory=dict)
    outputs: Dict[str, Dict[str, ParamAttributes]] = Field(default_factory=dict)
    constraints_in_parameters: Union[Dict[str, List[InterParamConstraint]], List[InterParamConstraint]] = Field(
        default_factory=dict
    )
    return_info: List[Any] = Field(default_factory=list)
    dtype_support_description: Union[Dict[str, Any], List[Any]] = Field(default_factory=dict)
    format_support_description: Union[Dict[str, Any], List[Any]] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


__all__ = [
    "ValueWithSrcText",
    "ParamAttributes",
    "InterConstraintsRuleType",
    "InterParamConstraint",
    "OperatorRule",
]
