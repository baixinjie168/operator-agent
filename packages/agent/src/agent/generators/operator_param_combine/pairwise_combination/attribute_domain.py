from __future__ import annotations

import copy
from typing import Any, Dict, List

from agent.generators.common_model_definition import OperatorRule, ParamAttributes, ValueWithSrcText
from agent.generators.common_utils.data_handle_utils import DataHandleUtil
from agent.generators.common_utils.logger_util import LazyLogger
from agent.generators.data_definition.constants import DataMatchMap, ParamModelConfig

logger = LazyLogger()

ATTR_TYPE = "type"
ATTR_FORMAT = "format"
ATTR_DTYPE = "dtype"
ATTR_DIMENSIONS = "dimensions"
ATTR_ARRAY_LENGTH = "array_length"
ATTR_RANGE_VALUE = "allowed_range_value"
ATTR_RANGE_VALUE_TYPE = "allowed_range_value_type"
ATTR_IS_OPTIONAL = "is_optional"
ATTR_IS_OPERATOR_PARAM = "is_operator_param"


class AttributeDomain:
    def __init__(self, operator_rule: OperatorRule):
        self.operator_rule = operator_rule
        self.param_domains: Dict[str, Dict[str, List[Any]]] = {}
        self._flatten_inputs()
        self._extract_all_domains()

    @staticmethod
    def _flatten_params(src):
        flat = {}
        for param_name, param_attr in src.items():
            if param_attr is None:
                continue
            if isinstance(param_attr, ParamAttributes):
                flat[param_name] = param_attr
            elif isinstance(param_attr, dict):
                platform_keys = list(param_attr.keys())
                if not platform_keys:
                    continue
                first_val = param_attr[platform_keys[0]]
                if isinstance(first_val, ParamAttributes):
                    flat[param_name] = first_val
                elif isinstance(first_val, dict):
                    flat[param_name] = ParamAttributes(**first_val)
                else:
                    flat[param_name] = ParamAttributes(**param_attr)
            else:
                flat[param_name] = param_attr
        return flat

    def _flatten_inputs(self):
        flat = self._flatten_params(self.operator_rule.inputs)
        flat.update(self._flatten_params(self.operator_rule.outputs))
        self.operator_rule.inputs = flat

    def _extract_all_domains(self):
        for param_name, param_attr in self.operator_rule.inputs.items():
            if param_attr is None:
                continue
            domain = {}
            domain[ATTR_TYPE] = self._extract_type_domain(param_name, param_attr)
            domain[ATTR_FORMAT] = self._extract_format_domain(param_name, param_attr)
            domain[ATTR_DTYPE] = self._extract_dtype_domain(param_name, param_attr)
            domain[ATTR_DIMENSIONS] = self._extract_dimensions_domain(param_name, param_attr)
            domain[ATTR_ARRAY_LENGTH] = self._extract_array_length_domain(param_name, param_attr)
            domain[ATTR_RANGE_VALUE], domain[ATTR_RANGE_VALUE_TYPE] = self._extract_range_value_domain(param_name, param_attr)
            domain[ATTR_IS_OPTIONAL] = self._extract_bool_domain(param_name, param_attr, "is_optional")
            domain[ATTR_IS_OPERATOR_PARAM] = self._extract_bool_domain(param_name, param_attr, "is_operator_param")
            self.param_domains[param_name] = domain

    @staticmethod
    def _get_value(param_name: str, attr: ValueWithSrcText | str, attr_name: str):
        raw, value_type = DataHandleUtil.get_relevant_attribute_value(param_name, attr, attr_name)
        return raw, value_type

    def _extract_type_domain(self, param_name: str, param_attr: ParamAttributes) -> List[str]:
        raw, _ = self._get_value(param_name, param_attr.type, "type")
        if raw is None:
            return [ParamModelConfig.DEFAULT_ATK_TYPE]
        if isinstance(raw, list):
            results = set()
            for r in raw:
                mapped = DataMatchMap.ACL_TYPE_TRANSFER_ATK_MAP.get(r, ParamModelConfig.DEFAULT_ATK_TYPE)
                results.add(mapped)
            return list(results)
        mapped = DataMatchMap.ACL_TYPE_TRANSFER_ATK_MAP.get(raw, ParamModelConfig.DEFAULT_ATK_TYPE)
        return [mapped]

    def _extract_format_domain(self, param_name: str, param_attr: ParamAttributes) -> List[str]:
        raw, _ = self._get_value(param_name, param_attr.format, "format")
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(v) for v in raw if v is not None]
        return [str(raw)]

    def _extract_dtype_domain(self, param_name: str, param_attr: ParamAttributes) -> List[str]:
        raw, _ = self._get_value(param_name, param_attr.dtype, "dtype")
        if raw is None:
            return [ParamModelConfig.DEFAULT_PARAM_DTYPE_DTYPE_IN_ORIGINAL_DOC]
        if isinstance(raw, list):
            return [str(v) for v in raw if v is not None]
        return [str(raw)]

    def _extract_dimensions_domain(self, param_name: str, param_attr: ParamAttributes) -> List[int]:
        """JSON中，dimension.value默认一定为枚举形式，即[2,6]表示dim可取值为2或6"""
        raw, _ = self._get_value(param_name, param_attr.dimensions, "dimensions")
        if raw is None:
            return list(range(
                ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM_MIN,
                ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM_MAX + 1
            ))
        if isinstance(raw, int):
            return [raw]
        if isinstance(raw, list):
            return raw
        return [ParamModelConfig.DEFAULT_TENSOR_SHAPE_DIM]

    def _extract_array_length_domain(self, param_name: str, param_attr: ParamAttributes) -> List[int]:
        raw, _ = self._get_value(param_name, param_attr.array_length, "array_length")
        if raw is None:
            return [ParamModelConfig.DEFAULT_LIST_LENGTH]
        if isinstance(raw, int):
            return [raw]
        if isinstance(raw, list):
            vals = []
            for item in raw:
                if isinstance(item, int):
                    vals.append(item)
                elif isinstance(item, list) and len(item) == 2 and all(isinstance(v, int) for v in item):
                    vals.extend(range(item[0], item[1] + 1))
            return sorted(set(vals)) if vals else [ParamModelConfig.DEFAULT_LIST_LENGTH]
        return [ParamModelConfig.DEFAULT_LIST_LENGTH]

    def _extract_range_value_domain(self, param_name: str, param_attr: ParamAttributes) -> tuple[List[Any], str|None ]:
        raw, value_type = self._get_value(param_name, param_attr.allowed_range_value, "allowed_range_value")
        if raw is None:
            return [], None
        if isinstance(raw, list):
            return copy.deepcopy(raw), value_type
        return [raw], value_type

    @staticmethod
    def _extract_bool_domain(param_name: str, param_attr: ParamAttributes, attr_name: str) -> List[bool]:
        raw, _ = DataHandleUtil.get_relevant_attribute_value(param_name, getattr(param_attr, attr_name), attr_name)
        if raw is None:
            return [True, False]
        if isinstance(raw, bool):
            return [raw]
        return [True, False]

    def get_dim_value_profile_domain(self) -> List[str]:
        return list(ParamModelConfig.DIM_VALUE_PROFILE_LIST)

    def get_param_type(self, param_name: str) -> str:
        domain = self.param_domains.get(param_name, {})
        type_list = domain.get(ATTR_TYPE, [ParamModelConfig.DEFAULT_ATK_TYPE])
        return type_list[0] if type_list else ParamModelConfig.DEFAULT_ATK_TYPE

    def is_tensor_param(self, param_name: str) -> bool:
        ptype = self.get_param_type(param_name)
        return ptype in ParamModelConfig.TENSOR_ATK_TYPE

    def is_list_param(self, param_name: str) -> bool:
        ptype = self.get_param_type(param_name)
        return ptype in ParamModelConfig.LIST_ATK_TYPE

    def get_dtype_combinations(self) -> List[Dict[str, str]] | None:
        dtype_desc = self.operator_rule.dtype_support_description
        if dtype_desc and isinstance(dtype_desc, list):
            return dtype_desc
        return None

    def get_effective_params(self) -> List[str]:
        return list(self.param_domains.keys())
