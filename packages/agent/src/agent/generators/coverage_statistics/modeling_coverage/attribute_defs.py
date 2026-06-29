"""
参数属性可能取值定义模块

功能：从算子约束 JSON 文件中提取每个参数的属性可能取值全集（Domain），
      为后续覆盖率计算提供"分母"（可能取值/可能组合的总空间）。

关键逻辑：
  1. 读取算子约束 JSON（OperatorRule 的原始格式）
  2. 针对每个参数，提取 dtype / format / dim_count / length 的允许取值
  3. dim_value_profile 使用全局离散化模型列表（固定 4 种策略）
  4. range_value_profile 根据参数 dtype 选择对应的浮点/整数/布尔离散化模型
"""

import json
import os
from typing import Dict, List, Any, Optional

from common_utils.logger_util import LazyLogger

logger = LazyLogger()


# ============================================================
# 全局离散化模型定义
# 对应 data_definition/constants.py 中 ParamModelConfig 的取值
# ============================================================

# shape 取值策略的离散化模型（4 种可枚举策略）
DIM_VALUE_PROFILE_LIST = ["Has_Large_Size", "Has_Size_1", "Has_Odd_Size", "Typical"]

# 浮点类型参数的数值范围离散化模型（9 种）
FLOAT_TENSOR_DATA_PROFILE = ["Typical", "PosNormal", "NegNormal", "Zero", "One", "NaN", "PosInf", "NegInf", "SubNormal"]

# 整数类型参数的数值范围离散化模型（6 种）
INT_TENSOR_DATA_PROFILE = ["Pos", "Neg", "Zero", "One", "Max", "Min"]

# bool 类型参数的取值
BOOL_DATA_PROFILE = [True, False]

# format 属性全局可能取值（来自 ParamModelConfig.FORMAT_VALUE_LIST）
FORMAT_VALUE_LIST = ["nchw", "nhwc", "nc", "cn", "fractal_nz", "nchw16", "nchw8", "chwn8", "nhwc8", "nhwc16"]

# 当算子约束 JSON 中未定义 dimensions 时，dim_count 的默认范围
DEFAULT_DIM_COUNT_MIN = 1
DEFAULT_DIM_COUNT_MAX = 8


NOT_RELEVANT_PARAM_VALUE = "N/A"


def _extract_value(raw_value: Any) -> Any:
    """
    从算子约束 JSON 的字段中提取实际值。
    算子约束 JSON 中字段格式可能为 {"value": ...} 的包装结构，
    也可能是直接量，此函数做统一解包。
    """
    if isinstance(raw_value, dict) and "value" in raw_value:
        return raw_value["value"]
    return raw_value


def _get_platform_data(raw_attr: dict) -> dict:
    """
    从算子约束 JSON 的平台嵌套结构中提取实际的参数属性数据。

    算子约束 JSON 中 inputs/outputs 的结构为：
      "paramName": {
          "Atlas 350 加速卡": { "type": {"value": "aclTensor"}, ... },
          "common": { ... }
      }

    本函数优先取 "common" 平台，否则取第一个可用的平台数据。
    如果已经是扁平结构（无平台层），则直接返回原值。
    """
    if not isinstance(raw_attr, dict):
        return {}

    platform_keys = [k for k in raw_attr.keys()
                     if isinstance(k, str) and (k.startswith("Atlas") or k == "common")]
    if not platform_keys:
        return raw_attr

    key = "common" if "common" in raw_attr else platform_keys[0]
    return raw_attr.get(key, {})


class ParamAttributeDomain:
    """
    单个参数的所有属性可能取值定义。

    每个参数有 6 个可枚举属性（部分属性因参数类型不同而存在/不存在）：
      - dtype: 数据类型（如 fp16, fp32, int8...）
      - format: 数据格式（如 ND, fractal_nz...）
      - dim_count: shape 维度数量（仅 tensor 参数）
      - dim_value_profile: shape 取值策略（仅 tensor 参数，固定 4 种离散模型）
      - range_value_profile: 数值范围模型（根据 dtype 确定浮点/整数/布尔模型）
      - length: 数组长度（仅 list 类型参数）
    """

    def __init__(self, param_name: str, param_type: str):
        self.param_name = param_name
        # param_type 取值为 "tensor" / "scalar" / "list" / "attr" 等（ATK 类型）
        self.param_type = param_type
        # 是否为真正的算子参数（从约束 JSON is_operator_param 字段读取）
        self.is_operator_param: bool = False

        # 以下列表将从算子约束 JSON 中填充
        self.dtype_values: List[str] = []
        self.format_values: List[str] = []
        self.dim_count_values: List[int] = []
        # dim_value_profile 固定为 4 种离散化模型，不依赖算子约束 JSON
        self.dim_value_profile_values: List[str] = list(DIM_VALUE_PROFILE_LIST)
        # range_value_profile 后续根据 dtype 选择对应的离散化模型
        self.range_value_profile_values: List[str] = []
        self.length_values: List[int] = []

    def is_tensor(self) -> bool:
        """判断是否为 tensor 类型参数（只有 tensor 才有 shape 相关属性）"""
        return self.param_type in ("tensor", "tensors")

    def is_list_type(self) -> bool:
        """判断是否为 list 类型参数（只有 list 才有 length 属性）"""
        return self.param_type in ("tensors", "scalars", "attrs")


class OperatorAttributeDomain:
    """
    单个算子的所有参数属性可能取值的全集。

    作为覆盖率计算的"分母"，记录该算子下每个参数的每个属性
    理论上可能取到的所有值范围。
    """

    def __init__(self, operator_name: str):
        self.operator_name = operator_name
        # key: 参数名, value: ParamAttributeDomain
        self.params: Dict[str, ParamAttributeDomain] = {}

    def get_param(self, name: str) -> Optional[ParamAttributeDomain]:
        return self.params.get(name)


def _get_range_profile_by_dtype(dtype: str) -> List[str]:
    """
    根据 dtype 选择对应的数值范围离散化模型。

    核心逻辑：
      1. 将 ACL dtype（如 FLOAT16, INT8）映射为 ATK dtype（如 fp16, int8）
      2. 判断 ATK dtype 属于浮点/整数/布尔中的哪一类
      3. 返回对应的离散化模型列表

    对应关系（来自 ParamModelConfig）：
      - 浮点类型 -> FLOAT_TENSOR_DATA_PROFILE（9 种）
      - 整数类型 -> INT_TENSOR_DATA_PROFILE（6 种）
      - 布尔类型 -> [True, False]
    """
    # ACL dtype -> ATK dtype 映射表
    acl_to_atk = {
        "INT4": "int", "INT8": "int8", "INT16": "int16", "INT32": "int32",
        "UINT8": "uint8", "UINT16": "uint16", "UINT32": "uint32", "UINT64": "uint64",
        "INT64": "int64", "BFLOAT16": "bf16", "FLOAT16": "fp16", "FLOAT32": "fp32",
        "FLOAT64": "fp64", "float32": "fp32", "float16": "fp16", "float64": "fp64",
        "FLOAT": "fp32", "DOUBLE": "double", "BOOL": "bool", "bool": "bool",
        "float": "fp32", "int64_t": "int",
    }
    atk_dtype = acl_to_atk.get(dtype, dtype)

    # 数据类型分类判断
    float_types = {"fp16", "fp32", "fp64", "bf16", "fp", "double"}
    int_types = {"int", "int16", "int8", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    bool_types = {"bool"}

    if atk_dtype in float_types:
        return list(FLOAT_TENSOR_DATA_PROFILE)
    elif atk_dtype in int_types:
        return list(INT_TENSOR_DATA_PROFILE)
    elif atk_dtype in bool_types:
        return [True, False]
    return []


def _resolve_dim_list(raw_dims: Any) -> List[int]:
    """
    解析算子约束 JSON 中的 dimensions/array_length 字段为取值列表。

    dimensions 字段（新枚举格式）：
      - [1, 2, 3, 4]        -> 直接枚举
      - [0, 3, 4]           -> 离散枚举（0 维 / 3 维 / 4 维）
      - None                 -> 使用默认范围 1~8
      - "N/A"                -> 不适用，返回空列表

    array_length 字段（仍用区间格式，未迁移）：
      - [[1, 4], [2, 3]]    -> 区间 [min, max] 展开
      - [128]               -> 单值

    注意：此函数被 dimensions 和 array_length 共用，区间展开分支
    必须保留以兼容 array_length 的 [[min, max]] 格式。
    """
    if raw_dims is None:
        return list(range(DEFAULT_DIM_COUNT_MIN, DEFAULT_DIM_COUNT_MAX + 1))

    if raw_dims == NOT_RELEVANT_PARAM_VALUE:
        return []

    if isinstance(raw_dims, list):
        result = []
        for d in raw_dims:
            if isinstance(d, list) and len(d) >= 2:  # 区间展开（array_length 仍用此格式）
                result.extend(range(d[0], d[1] + 1))
            elif isinstance(d, (int, float)):  # 枚举值（dimensions 新格式走此分支）
                result.append(int(d))
        return sorted(set(result)) if result else [DEFAULT_DIM_COUNT_MIN]

    return [DEFAULT_DIM_COUNT_MIN]


def _resolve_list(raw: Any, ignore_na: bool = True) -> List:
    """
    将算子约束 JSON 中可能以多种格式存储的列表字段统一解析为 Python list。

    处理三种情况：
      - None -> []
      - 已经是 list -> 原样返回
      - {"value": [...]} 包装结构 -> 解包

    如果 ignore_na 为 True，则会过滤掉 "N/A"（NOT_RELEVANT_PARAM_VALUE）。
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        result = raw
    elif isinstance(raw, dict) and "value" in raw:
        val = raw["value"]
        result = val if isinstance(val, list) else [val]
    else:
        result = [raw]
    if ignore_na:
        result = [v for v in result if v != NOT_RELEVANT_PARAM_VALUE]
    return result


# ACL 类型到 ATK 类型的映射（来自 DataMatchMap.ACL_TYPE_TRANSFER_ATK_MAP）
ATK_TYPE_MAP = {
    "aclTensor": "tensor", "aclScalar": "scalar", "aclIntArray": "attrs",
    "aclFloatArray": "attrs", "aclBoolArray": "attrs", "aclTensorList": "tensors",
    "aclScalarList": "scalars",
}


def _build_domain_for_params(
    domain: OperatorAttributeDomain,
    params_dict: dict,
    source_label: str,
):
    """
    从算子约束 JSON 的 params_dict（inputs 或 outputs 部分）构建参数属性域。

    每个参数的结构为平台嵌套格式，需要先通过 _get_platform_data 解包。

    Args:
        domain: 待填充的 OperatorAttributeDomain
        params_dict: 约束 JSON 中的 inputs 或 outputs 字典
        source_label: 来源标识（"inputs" 或 "outputs"），仅用于日志
    """
    for param_name, raw_attr in params_dict.items():
        # --- 解包平台嵌套结构 ---
        attr_data = _get_platform_data(raw_attr)
        if not attr_data:
            logger.debug(f"  [{param_name}] no platform data found, skipped")
            continue

        # --- 步骤 1: 提取参数类型并映射为 ATK 类型 ---
        raw_type = _extract_value(attr_data.get("type"))
        param_type = ATK_TYPE_MAP.get(raw_type, "attr")

        # --- 步骤 2: 为当前参数创建域定义 ---
        p = ParamAttributeDomain(param_name, param_type)

        # --- 标记是否为真正的算子参数 ---
        p.is_operator_param = bool(_extract_value(attr_data.get("is_operator_param")))

        # --- 步骤 3: 提取各个属性的允许取值 ---
        # dtype: 可能取列表（如 ["FLOAT16", "BFLOAT16", "FLOAT32"]），需映射为 ATK 名称
        raw_dtype_list = _resolve_list(_extract_value(attr_data.get("dtype")))
        acl_to_atk = {
            "INT4": "int", "INT8": "int8", "INT16": "int16", "INT32": "int32",
            "UINT8": "uint8", "UINT16": "uint16", "UINT32": "uint32", "UINT64": "uint64",
            "INT64": "int64", "BFLOAT16": "bf16", "FLOAT16": "fp16", "FLOAT32": "fp32",
            "FLOAT64": "fp64", "float32": "fp32", "float16": "fp16", "float64": "fp64",
            "FLOAT": "fp32", "DOUBLE": "double", "BOOL": "bool", "bool": "bool",
            "float": "fp32", "int64_t": "int",
        }
        p.dtype_values = [acl_to_atk.get(d, d) for d in raw_dtype_list]
        logger.debug(f"  [{param_name}] dtype domain: {p.dtype_values}")

        # format: 可能取列表（如 ["ND", "NCHW"]），过滤 "N/A"
        p.format_values = _resolve_list(_extract_value(attr_data.get("format")))
        logger.debug(f"  [{param_name}] format domain: {p.format_values}")

        # dim_count: 可能取列表或区间（如 [1, 2, 3, 4] 或 [[1, 4]]）
        p.dim_count_values = _resolve_dim_list(_extract_value(attr_data.get("dimensions")))
        logger.debug(f"  [{param_name}] dim_count domain: {p.dim_count_values}")

        # length: 仅 list 类型参数有（格式同 dimensions：支持 [[min, max]] 区间）
        p.length_values = _resolve_dim_list(_extract_value(attr_data.get("array_length")))
        if p.length_values:
            logger.debug(f"  [{param_name}] length domain: {p.length_values}")

        # --- 步骤 4: 根据 dtype 确定 range_value 的离散化模型 ---
        # 取第一个 dtype 作为代表（所有 dtype 同属一类：浮点/整数/布尔）
        if p.dtype_values:
            p.range_value_profile_values = _get_range_profile_by_dtype(p.dtype_values[0])
            logger.debug(f"  [{param_name}] range_value_profile domain: {p.range_value_profile_values} "
                         f"(derived from dtype: {p.dtype_values[0]})")

        # --- 步骤 5: 注册参数 ---
        domain.params[param_name] = p


def build_domain_from_constraint(constraint_json_path: str) -> OperatorAttributeDomain:
    """
    从算子约束 JSON 文件构建完整的属性域定义。

    处理流程：
       1. 读取 JSON 文件
       2. 遍历 inputs 和 outputs 中的每个参数（outputs 中的参数也会被加入到
          case 的 inputs 中，因此也需要构建域定义）
       3. 解包平台嵌套结构，提取 type/dtype/format/dimensions/array_length 字段
       4. 根据 dtype 确定 range_value_profile 的离散化模型
       5. 组装为 ParamAttributeDomain 并注册到 OperatorAttributeDomain

    Args:
        constraint_json_path: 算子约束 JSON 文件路径

    Returns:
        OperatorAttributeDomain: 包含所有参数属性可能取值的域定义
    """
    logger.info(f"Building attribute domain from constraint file: {constraint_json_path}")
    with open(constraint_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    operator_name = data.get(
        "operator_name",
        os.path.splitext(os.path.basename(constraint_json_path))[0]
    )
    domain = OperatorAttributeDomain(operator_name)

    # 从 inputs 构建域
    inputs = data.get("inputs", {})
    logger.info(f"Operator: {operator_name}, input params in constraint: {len(inputs)}")
    _build_domain_for_params(domain, inputs, "inputs")

    # 从 outputs 构建域（outputs 中的参数也会出现在 case 的 inputs 中）
    outputs = data.get("outputs", {})
    if outputs:
        new_outputs = {k: v for k, v in outputs.items() if k not in domain.params}
        if new_outputs:
            logger.info(f"  Also processing {len(new_outputs)} output params not in inputs: {list(new_outputs.keys())}")
            _build_domain_for_params(domain, new_outputs, "outputs")

    logger.info(f"Domain built: operator={operator_name}, params={len(domain.params)}")
    return domain
