#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATK API 脚本生成器

输入: ATK 用例 JSON + aclnn.txt 签名表
输出: 可被 ATK 加载的 .py 文件，能调用 ACLNN 算子并返回结果

用法:
    python generator.py <case_json> [-o output.py] [--signatures aclnn.txt]
"""

import argparse
import copy
import json
import os
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# C++ 签名解析
# ---------------------------------------------------------------------------

# C++ 签名参数类型 -> ATK JSON input type 映射
CPP_TYPE_TO_JSON_TYPE = {
    "aclTensor*": "tensor",
    "const aclTensor*": "tensor",
    "const aclTensorList*": "tensorList",
    "aclTensorList*": "tensorList",
    "const aclScalar*": "scalar",
    "aclScalar*": "scalar",
    "const aclScalarList*": "scalarList",
    "aclScalarList*": "scalarList",
    # aclIntArray 类型
    "const aclIntArray*": "attr",
    "aclIntArray*": "attr",
    # 标量 / attr 类型
    "double": "attr",
    "float": "attr",
    "int8_t": "attr",
    "int32_t": "attr",
    "int64_t": "attr",
    "uint8_t": "attr",
    "uint32_t": "attr",
    "uint64_t": "attr",
    "bool": "attr",
    "attr_bool": "attr",
    "str": "attr",
    "const char*": "attr",
    "aclString*": "attr",
    "const aclString*": "attr",
    # 框架固定参数 (不需要从 JSON 取值)
    "uint64_t*": "framework",
    "aclOpExecutor**": "framework",
}

# 框架固定参数名 —— 由 ATK 后端自动追加，不需要用户在 JSON 中配置
FRAMEWORK_PARAMS = {"workspaceSize", "executor"}


def parse_cpp_signature(signature: str) -> list[dict]:
    """
    解析 C++ 签名字符串，返回参数列表。
    每个元素: {"name": str, "raw_type": str, "kind": str}
      kind: "tensor" | "scalar" | "attr" | "framework" | "output"
    """
    # 提取括号内内容
    match = re.search(r"\((.+)\)\s*$", signature, re.DOTALL)
    if not match:
        raise ValueError(f"无法解析签名: {signature}")

    body = match.group(1).strip()
    params = []
    # 按逗号分割，但要尊重括号嵌套
    depth = 0
    current = []
    for ch in body:
        if ch in ("(", "<"):
            depth += 1
            current.append(ch)
        elif ch in (")", ">"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        params.append("".join(current).strip())

    result = []
    for raw in params:
        raw = raw.strip()
        if not raw:
            continue
        # 匹配 C++ 参数声明，处理以下格式：
        #   const aclTensor* input       (type*name, no space)
        #   const aclTensor *input       (type *name, space before *)
        #   const aclTensor   *biasOpt   (type * name, spaces around *)
        #   aclOpExecutor **executor     (double pointer)
        #   double eps                   (plain type, no pointer)
        #   const aclIntArray *shape     (aclIntArray pointer)
        # 策略：先从末尾提取参数名，剩余部分整理为类型字符串
        # 先从末尾提取参数名（最后一个单词）
        name_m = re.search(r'(\w+)\s*$', raw)
        if not name_m:
            continue
        param_name = name_m.group(1)
        # 去掉参数名后的剩余部分即为类型（可能包含 * 和空格）
        type_raw = raw[:name_m.start()].strip()
        # 清理类型字符串中的多余空格，保留 * 和 **
        # 去掉尾部的 * 符号（可能有多个），再拼回去
        stars = ""
        while type_raw.endswith("*"):
            stars = "*" + stars
            type_raw = type_raw[:-1].strip()
        # 清理类型部分的内部多余空格
        type_clean = re.sub(r'\s+', ' ', type_raw).strip()
        type_str = type_clean + stars

        # 查映射
        kind = CPP_TYPE_TO_JSON_TYPE.get(type_str, "attr")

        # 判断是否是输出参数: 非 const 的 aclTensor*
        # 例外: selfRef / inputRef 是 inplace 输入（既读又写），不算纯 output
        INPLACE_NAMES = {"selfRef", "inputRef", "self", "input"}
        if type_str == "aclTensor*" and param_name not in FRAMEWORK_PARAMS:
            if param_name in INPLACE_NAMES:
                kind = "tensor"  # inplace input, not output
            else:
                kind = "output"
        elif param_name in FRAMEWORK_PARAMS:
            kind = "framework"

        result.append({
            "name": param_name,
            "raw_type": type_str,
            "kind": kind,
        })

    return result


# ---------------------------------------------------------------------------
# 签名表加载
# ---------------------------------------------------------------------------

def load_signatures(path: str) -> dict[str, str]:
    """
    加载 aclnn.txt，返回 {op_name: signature} 字典。
    格式: 每行 "算子名 签名字符串"
    """
    sigs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 格式: 算子名 签名(...)
            # 签名以 aclnnStatus 开头，以 ) 结尾，后面可能有多余内容
            m = re.match(r'^(\S+)\s+(aclnnStatus\s+.*?\))', line)
            if m:
                op_name = m.group(1)
                signature = m.group(2).strip()
                sigs[op_name] = signature
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    sigs[parts[0]] = parts[1].strip()
                elif len(parts) == 1:
                    sigs[parts[0]] = parts[0]
    return sigs


# ---------------------------------------------------------------------------
# 代码生成
# ---------------------------------------------------------------------------

def _extract_op_name(aclnn_name: str) -> str:
    """规范化算子名，确保以 aclnn 开头"""
    if aclnn_name and aclnn_name.startswith("aclnn"):
        return aclnn_name
    return "aclnn" + (aclnn_name or "")


def _collect_api_types(cases: list[dict]) -> tuple[list[str], str]:
    """
    收集所有用例中的 api_type 和 aclnn_api_type 取值。
    返回 (api_types_list, aclnn_api_type) — api_type 可能有多个不同取值
    """
    api_types = []
    aclnn_api_type = ""
    for case in cases:
        t = case.get("api_type", "")
        if t and t not in api_types:
            api_types.append(t)
        if not aclnn_api_type:
            t2 = case.get("aclnn_api_type", "")
            if t2:
                aclnn_api_type = t2
    return api_types, aclnn_api_type


def _collect_all_inputs(cases: list[dict]) -> list[dict]:
    """收集所有用例中出现的 inputs，去重后返回"""
    seen = set()
    all_inputs = []
    for case in cases:
        for inp in case.get("inputs", []):
            # normalizedShape 是嵌套列表格式，需展平处理
            if isinstance(inp, list):
                for sub in inp:
                    if isinstance(sub, dict):
                        name = sub.get("name", "")
                        if name and name not in seen:
                            seen.add(name)
                            all_inputs.append(sub)
                continue
            name = inp.get("name", "")
            if name and name not in seen:
                seen.add(name)
                all_inputs.append(inp)
    return all_inputs


def _has_named_inputs(cases: list[dict]) -> bool:
    """
    检查 JSON 用例中 inputs 是否有非空的 name 字段。
    如果有 name → ATK 使用 kwargs 传参；如果全为空 → ATK 使用 args 传参。
    """
    for case in cases:
        for inp in case.get("inputs", []):
            if isinstance(inp, list):
                for sub in inp:
                    if isinstance(sub, dict) and sub.get("name", ""):
                        return True
            elif inp.get("name", ""):
                return True
    return False


# ---------------------------------------------------------------------------
# NZ (FRACTAL_NZ) 格式检测与辅助
# ---------------------------------------------------------------------------

def _flatten_shape(shape):
    """将可能嵌套的 shape 列表展平为一维 list。"""
    if shape is None:
        return []
    result = []
    if isinstance(shape, (list, tuple)):
        for s in shape:
            if isinstance(s, (list, tuple)):
                result.extend(_flatten_shape(s))
            else:
                result.append(s)
    else:
        result.append(shape)
    return result


def _inverse_perm(perm):
    """计算排列的逆排列。"""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _nz_forward_perm(batch_dims):
    """NZ 正向排列索引（稠密 reshape → NZ 内存布局）。

    对于最后 4 维 [k1, k0, n1, n0]，正向排列 [2, 0, 1, 3] 将其变为
    [n1, k1, k0, n0]（NZ 存储顺序），batch 维度保持不变。
    """
    return list(range(batch_dims)) + [batch_dims + i for i in [2, 0, 1, 3]]


def _nz_inverse_perm(batch_dims):
    """NZ 逆排列索引（NZ → 稠密 reshape 前的排列）。"""
    return _inverse_perm(_nz_forward_perm(batch_dims))


def _find_k_source(nz_param, json_inputs, sig_params):
    """
    确定哪个输入张量提供 Reduce 维度 k。
    返回 (k_source_name, k_source_axis) 或 (None, None)。

    - AddmmWeightNz: out = beta*self + alpha*(mat1 @ mat2)，k_source = mat1，轴 -1
    - MatmulWeightNz / BatchMatMulWeightNz: out = self @ mat2，k_source = self
      (batch 3D → axis -1；非 batch 2D → axis -2)
    - 默认：第一个不是 NZ 张量的其他 tensor 输入
    """
    nz_name = nz_param.get("name", "")
    nz_dims = nz_param.get("nz_dims", 4)
    sig_names = {p["name"] for p in sig_params}

    # AddmmWeightNz: mat1 @ mat2，k 来自 mat1 的最后一维
    if "mat1" in sig_names and nz_name == "mat2":
        return "mat1", -1

    if "self" in sig_names:
        if nz_dims >= 5:
            return "self", -1
        return "self", -2

    for inp in json_inputs:
        if not isinstance(inp, dict):
            continue
        iname = inp.get("name", "")
        if not iname or iname == nz_name:
            continue
        if (inp.get("format") or "").upper() == "NZ":
            continue
        if inp.get("type") in ("tensor", "tensors"):
            return iname, -1
    return None, None


def _detect_nz_params(cases, sig_params):
    """
    检测 NZ 格式张量参数，返回描述 NZ 参数的字典列表。

    检测方式:
      1. JSON 输入定义中包含 "format": "NZ"
      2. 算子名称包含 WeightNz（不区分大小写）且该张量是权重参数
         （名称为 mat2 / weight / B，或维度最多的张量）

    每个元素:
      {"name": str, "nz_dims": int, "batch_dims": int,
       "k_source": str|None, "k_source_axis": int|None}
    """
    json_inputs = _collect_all_inputs(cases)

    op_name = ""
    for case in cases:
        n = case.get("aclnn_name") or case.get("name", "")
        if n:
            op_name = n
            break

    nz_params = []

    # 方式 1: 显式 format == "NZ"
    for inp in json_inputs:
        if not isinstance(inp, dict):
            continue
        if (inp.get("format") or "").upper() != "NZ":
            continue
        flat_shape = _flatten_shape(inp.get("shape"))
        if len(flat_shape) < 4:
            continue
        nz_dims = len(flat_shape)
        batch_dims = max(nz_dims - 4, 0)
        nz_p = {
            "name": inp.get("name", ""),
            "nz_dims": nz_dims,
            "batch_dims": batch_dims,
        }
        k_src, k_axis = _find_k_source(nz_p, json_inputs, sig_params)
        nz_p["k_source"] = k_src
        nz_p["k_source_axis"] = k_axis
        nz_params.append(nz_p)

    # 方式 2: 算子名含 WeightNz 且未显式标记
    if not nz_params and op_name and "weightnz" in op_name.lower():
        weight_names = {"mat2", "weight", "B"}
        candidate = None
        for inp in json_inputs:
            if isinstance(inp, dict) and inp.get("name") in weight_names:
                candidate = inp
                break
        if candidate is None:
            best_dims = -1
            for inp in json_inputs:
                if not isinstance(inp, dict):
                    continue
                if inp.get("type") not in ("tensor", "tensors"):
                    continue
                flat_shape = _flatten_shape(inp.get("shape"))
                if len(flat_shape) > best_dims:
                    best_dims = len(flat_shape)
                    candidate = inp
        if candidate is not None:
            flat_shape = _flatten_shape(candidate.get("shape"))
            nz_dims = len(flat_shape)
            batch_dims = max(nz_dims - 4, 0)
            nz_p = {
                "name": candidate.get("name", ""),
                "nz_dims": nz_dims,
                "batch_dims": batch_dims,
            }
            k_src, k_axis = _find_k_source(nz_p, json_inputs, sig_params)
            nz_p["k_source"] = k_src
            nz_p["k_source_axis"] = k_axis
            nz_params.append(nz_p)

    return nz_params


def _detect_comm_params(sig_params):
    """
    检测通信域参数（const char* group/groupEp/groupTp），返回参数名列表。

    ACLNN 分布式算子的通信域参数需要传入 HCCL 通信域名称字符串（commName），
    但 ATK 在 JSON 中可能传入 PyTorch ProcessGroup 对象或任意值。
    生成器自动生成代码：从分布式上下文获取 commName，不依赖 JSON 传入值。

    检测条件: 参数名以 "group" 开头（不区分大小写）且 raw_type 包含 "char*"
    覆盖: group（单通信域）、groupEp（专家并行）、groupTp（张量并行）
    """
    comm_params = []
    for p in sig_params:
        name = p.get("name", "")
        raw_type = p.get("raw_type", "")
        if name.lower().startswith("group") and "char" in raw_type:
            comm_params.append(name)
    return comm_params


def generate_api_class_for_op(cases: list[dict], signature: str, op_name: str) -> str:
    """
    为同一个算子的所有用例生成一个通用的 ATK API py 文件。

    生成两个类:
    1. AclnnBaseApi 子类 — 给 PyAclnn 后端调用算子用 (aclnn_api_type)
    2. BaseApi 子类 — 给 CPU 后端用，返回 dummy 结果让 ATK 流程跑通 (api_type)

    关键: 收集所有签名中的非 framework/output 参数名，
    在 init_by_input_data 中运行时判断哪些参数 JSON 没传，动态补 NULL。
    """
    sig_params = parse_cpp_signature(signature)

    # 从 JSON outputs 字段读取输出参数名，覆盖签名解析结果
    # 某些算子签名中所有 tensor 都是 const aclTensor*，无法靠 const 区分输入/输出
    # 此时 JSON 中指定: "outputs": ["yOut", "meanOut", "rstdOut", "xOut"]
    output_names = set()
    for _case in cases:
        _outputs = _case.get("outputs")
        if _outputs:
            if isinstance(_outputs, list):
                output_names.update(_outputs)
            elif isinstance(_outputs, str):
                # 支持逗号分隔: "yOut,meanOut,rstdOut,xOut"
                output_names.update(n.strip() for n in _outputs.split(",") if n.strip())
    if output_names:
        for _p in sig_params:
            if _p["name"] in output_names:
                _p["kind"] = "output"

    # 签名完整参数顺序（含 output / framework），用于按签名顺序构建参数列表
    all_sig_param_info = [
        {"name": p["name"], "raw_type": p["raw_type"], "kind": p["kind"]}
        for p in sig_params
    ]

    user_sig_params = [
        p for p in sig_params
        if p["kind"] not in ("output", "framework")
        and p["name"] not in FRAMEWORK_PARAMS
    ]
    param_names = [p["name"] for p in user_sig_params]
    # 记录签名中所有非 framework 参数的 raw_type，用于为缺失的可选参数生成正确类型的 NULL
    param_type_map = {p["name"]: p["raw_type"] for p in sig_params if p["name"] not in FRAMEWORK_PARAMS}

    api_types, aclnn_api_type = _collect_api_types(cases)
    if not aclnn_api_type:
        aclnn_api_type = "pyaclnn_aclnn_" + op_name.lower()

    aclnn_class_name = _to_class_name(aclnn_api_type)

    # NZ 格式张量检测
    nz_params = _detect_nz_params(cases, sig_params)
    has_nz = bool(nz_params)

    # 通信域参数检测（const char* group）
    comm_params = _detect_comm_params(sig_params)
    has_comm = bool(comm_params)

    # 始终置空参数检测（如 alltoAllAxesOptional，文档说"传入空时默认按[-2,-1]处理"）
    _ALWAYS_NULL_NAMES = {"alltoAllAxesOptional"}
    always_null_params = [p["name"] for p in sig_params if p["name"] in _ALWAYS_NULL_NAMES]
    has_always_null = bool(always_null_params)

    lines = []
    lines.append("# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.")
    lines.append("# Auto-generated by generator.py")
    lines.append("")
    lines.append("import copy")
    lines.append("import ctypes")
    if has_nz:
        lines.append("import os")
    lines.append("")
    lines.append("import torch")
    if has_nz:
        lines.append("import torch.nn.functional as F")
    lines.append('from atk.common.log import Logger')
    lines.append("from atk.configs.dataset_config import InputDataset")
    lines.append("from atk.configs.results_config import TaskResult")
    lines.append("from atk.tasks.api_execute import register")
    lines.append("from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi")
    lines.append("from atk.tasks.api_execute.base_api import BaseApi")
    if has_nz:
        lines.append("from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclTensorlistStruct, AclFormat, nnopbase")
        lines.append('os.environ["PYTORCH_NO_NPU_MEMORY_CACHING"] = "1"')
    else:
        lines.append("from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclTensorlistStruct")
    lines.append("")
    lines.append('logging = Logger().get_logger()')
    lines.append("")
    lines.append("")

    # ========== Class 1: AclnnBaseApi (pyaclnn 后端 - 调用算子拿结果) ==========
    lines.append(f'@register("{aclnn_api_type}")')
    lines.append(f"class {aclnn_class_name}(AclnnBaseApi):")
    lines.append(f'    """Auto-generated API class for {op_name} (pyaclnn backend)."""')
    lines.append("")
    lines.append("    # 签名完整参数顺序（含 user / output，不含 framework），用于按签名顺序构建参数列表")
    lines.append(f"    _SIG_ORDER = {all_sig_param_info}")
    lines.append("    # 每个签名参数对应的 C++ 原始类型，用于为缺失的可选参数生成正确类型的 NULL")
    lines.append(f"    _SIG_PARAM_TYPES = {param_type_map}")
    lines.append("")
    lines.append("    # 签名 attr 标量 raw_type → ctypes 类型映射，用于校正 ATK convert_input_data 的类型")
    lines.append("    _ATTR_TYPE_MAP = {")
    lines.append("        'double': ctypes.c_double, 'float': ctypes.c_float,")
    lines.append("        'int8_t': ctypes.c_int8, 'int32_t': ctypes.c_int32,")
    lines.append("        'int64_t': ctypes.c_int64, 'uint8_t': ctypes.c_uint8,")
    lines.append("        'uint32_t': ctypes.c_uint32, 'uint64_t': ctypes.c_uint64,")
    lines.append("        'bool': ctypes.c_bool,")
    lines.append("    }")
    if has_always_null:
        lines.append(f"    _ALWAYS_NULL_PARAMS = {repr(set(always_null_params))}")
    lines.append("")
    lines.append("    def _get_null_for_param(self, name, raw_type):")
    lines.append("        \"\"\"为缺失的可选参数返回对应 ctypes 类型的 NULL，通过签名校验。\"\"\"")
    lines.append("        if 'aclTensorList' in raw_type:")
    lines.append("            from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorList")
    lines.append("            return ctypes.POINTER(AclTensorList)()")
    lines.append("        if 'aclScalarList' in raw_type:")
    lines.append("            from atk.tasks.backends.lib_interface.acl_wrapper import AclScalarList")
    lines.append("            return ctypes.POINTER(AclScalarList)()")
    lines.append("        if 'aclTensor' in raw_type:")
    lines.append("            from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor")
    lines.append("            return ctypes.POINTER(AclTensor)()")
    lines.append("        if 'aclScalar' in raw_type:")
    lines.append("            from atk.tasks.backends.lib_interface.acl_wrapper import AclScalar")
    lines.append("            return ctypes.POINTER(AclScalar)()")
    lines.append("        if 'aclIntArray' in raw_type:")
    lines.append("            from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray")
    lines.append("            return ctypes.POINTER(AclIntArray)()")
    lines.append("        # attr / 标量类型，根据 raw_type 转换为对应的 ctypes 零值")
    lines.append("        type_to_ctype = {")
    lines.append("            'double': ctypes.c_double, 'float': ctypes.c_float,")
    lines.append("            'int8_t': ctypes.c_int8, 'int32_t': ctypes.c_int32,")
    lines.append("            'int64_t': ctypes.c_int64, 'uint8_t': ctypes.c_uint8,")
    lines.append("            'uint32_t': ctypes.c_uint32, 'uint64_t': ctypes.c_uint64,")
    lines.append("            'bool': ctypes.c_bool,")
    lines.append("        }")
    lines.append("        ctype = type_to_ctype.get(raw_type, ctypes.c_void_p)")
    lines.append("        return ctype(0)")
    lines.append("")

    # _get_kwarg: 从 input_data 中按名称多模式查找参数（NZ 和非 NZ 路径共用）
    lines.append("    def _get_kwarg(self, input_data, name):")
    lines.append('        """从 input_data 中按名称多模式查找参数。"""')
    lines.append("        # Mode 1: kwargs 精确匹配")
    lines.append("        kwarg = input_data.kwargs.get(name)")
    lines.append("        # Mode 2: kwargs 不区分大小写匹配")
    lines.append("        if kwarg is None and input_data.kwargs:")
    lines.append("            _name_lower = name.lower()")
    lines.append("            for _k, _v in input_data.kwargs.items():")
    lines.append("                if isinstance(_k, str) and _k.lower() == _name_lower and _v is not None:")
    lines.append("                    kwarg = _v")
    lines.append("                    break")
    lines.append("        # Mode 3: args 字典搜索")
    lines.append("        if kwarg is None and input_data.args:")
    lines.append("            _name_lower = name.lower()")
    lines.append("            for _arg in input_data.args:")
    lines.append("                if isinstance(_arg, dict):")
    lines.append("                    _val = _arg.get(name)")
    lines.append("                    if _val is not None:")
    lines.append("                        kwarg = _val")
    lines.append("                        break")
    lines.append("                    for _ak, _av in _arg.items():")
    lines.append("                        if isinstance(_ak, str) and _ak.lower() == _name_lower and _av is not None:")
    lines.append("                            kwarg = _av")
    lines.append("                            break")
    lines.append("                if kwarg is not None:")
    lines.append("                    break")
    lines.append("        return kwarg")
    lines.append("")

    # get_format: 为 NZ 张量返回 ACL_FORMAT_FRACTAL_NZ
    if has_nz:
        lines.append("    def get_format(self, input_data: InputDataset, index=None, name=None):")
        for nz_p in nz_params:
            lines.append(f'        if name == "{nz_p["name"]}":')
            lines.append("            return AclFormat.ACL_FORMAT_FRACTAL_NZ")
        lines.append("        return AclFormat.ACL_FORMAT_ND")
        lines.append("")

    # get_storage_shape: 返回 NZ 存储形状（由 init_by_input_data 中的 _nz_storage_shape 设置）
    if has_nz:
        nz_names_repr = repr([nz_p["name"] for nz_p in nz_params])
        lines.append("    def get_storage_shape(self, input_data: InputDataset, index=None, name=None):")
        lines.append(f"        _nz_names = {nz_names_repr}")
        lines.append("        if name is not None and name in _nz_names and input_data.kwargs and name in input_data.kwargs:")
        lines.append("            t = input_data.kwargs[name]")
        lines.append("            if t.dim() >= 4:")
        lines.append("                # 返回转换后的 NZ storage shape（由 _nz_storage_shape 设置）")
        lines.append('                return getattr(self, "_nz_storage_shape", torch.Size(list(t.shape)))')
        lines.append("        elif name is not None and input_data.kwargs and name in input_data.kwargs:")
        lines.append("            return input_data.kwargs[name].shape")
        lines.append("        return None")
        lines.append("")

    lines.append("    def init_by_input_data(self, input_data: InputDataset):")
    if has_nz:
        lines.append("        import torch_npu")
    lines.append("        input_args = []")
    lines.append("        output_packages = []")
    lines.append("        _output_idx = 0  # 当前处理到第几个 output 参数")
    lines.append("")

    if has_comm:
        comm_names_repr = repr(comm_params)
        lines.append("        # 通信域参数处理：从分布式上下文获取 HCCL 通信域名称，不依赖 JSON 传入值")
        lines.append("        if hasattr(self, 'dist_task_info'):")
        lines.append("            import torch.distributed as dist")
        lines.append("            _rank_id = self.dist_task_info.rank")
        lines.append(f"            for _comm_param in {comm_names_repr}:")
        lines.append("                _pg = self._get_kwarg(input_data, _comm_param)")
        lines.append("                # JSON 传入的值可能不是 ProcessGroup（任意字符串/None），用 world group 兜底")
        lines.append("                if _pg is None or not hasattr(_pg, '_get_backend'):")
        lines.append("                    _pg = dist.group.WORLD")
        lines.append("                if _pg is not None:")
        lines.append("                    input_data.kwargs[_comm_param] = _pg._get_backend(torch.device(\"npu\")).get_hccl_comm_name(_rank_id)")
        lines.append("")

    if has_nz:
        lines.append("        # NZ 张量预处理：ND → 稠密 → NZ 内存布局")
        lines.append("        _nz_converted = {}")
        lines.append("        _self_converted = {}")
        for nz_p in nz_params:
            _nz_name = nz_p["name"]
            _k_src = nz_p["k_source"]
            _k_axis = nz_p["k_source_axis"]
            _bd = nz_p["batch_dims"]
            _inv_idx = _nz_inverse_perm(_bd)
            _fwd_idx = _nz_forward_perm(_bd)
            lines.append(f'        _nz_in = self._get_kwarg(input_data, "{_nz_name}")')
            if _k_src:
                lines.append(f'        _k_src = self._get_kwarg(input_data, "{_k_src}")')
            else:
                lines.append("        _k_src = None")
            lines.append('        _self_in = self._get_kwarg(input_data, "self")')
            lines.append('        _self_transposed = self._get_kwarg(input_data, "self_transposed") or False')
            lines.append('        _mat2_transposed = self._get_kwarg(input_data, "mat2_transposed") or False')
            lines.append("        if _self_in is not None and _self_transposed:")
            lines.append('            _self_converted["self"] = _self_in.transpose(1, 2)')
            lines.append("        if _nz_in is not None and _nz_in.dim() >= 4:")
            lines.append("            _bdims = list(_nz_in.shape[:-4])")
            lines.append("            if _mat2_transposed:")
            lines.append("                # 转置 NZ: (b, k1, n1, n0, k0) where n0=16, k0=16")
            lines.append("                _k1 = _nz_in.shape[-4]")
            lines.append("                _n1 = _nz_in.shape[-3]")
            lines.append("                _n0 = _nz_in.shape[-2]")
            lines.append("                _k0 = _nz_in.shape[-1]")
            lines.append("                _k = _k1 * _k0")
            lines.append("                _n = _n1 * _n0")
            lines.append(f"                _inv_idx_t = {_inv_idx}")
            lines.append("                _dense_padded = _nz_in.permute(_inv_idx_t).reshape(*_bdims, _k1 * _k0, _n1 * _n0)")
            lines.append("                _outer = _k1 * _k0")
            lines.append("                _inner = _n1 * _n0")
            lines.append("            else:")
            lines.append("                # 非转置 NZ: (b, n1, k1, k0, n0) where k0=16, n0=16")
            lines.append("                _n1 = _nz_in.shape[-4]")
            lines.append("                _k1 = _nz_in.shape[-3]")
            lines.append("                _k0 = _nz_in.shape[-2]")
            lines.append("                _n0 = _nz_in.shape[-1]")
            lines.append("                _k = _k1 * _k0")
            lines.append("                _n = _n1 * _n0")
            lines.append(f"                _inv_idx_nt = {_inv_idx}")
            lines.append("                _dense_padded = _nz_in.permute(_inv_idx_nt).reshape(*_bdims, _k1 * _k0, _n1 * _n0)")
            lines.append("                _outer = _k1 * _k0")
            lines.append("                _inner = _n1 * _n0")
            lines.append('            _out_t = self._get_kwarg(input_data, "out")')
            lines.append("            if _out_t is not None:")
            lines.append("                _n = _out_t.shape[-1]")
            lines.append("            _dense = _dense_padded[..., :_k, :_n]")
            lines.append("            _outer = ((_k + 15) // 16) * 16")
            lines.append("            _inner = ((_n + 15) // 16) * 16")
            lines.append("            _pad_t = F.pad(_dense, (0, _inner - _n, 0, _outer - _k))")
            lines.append("            _reshaped = _pad_t.reshape(*_bdims, _outer // 16, 16, _inner // 16, 16)")
            lines.append(f"            _fwd_idx = {_fwd_idx}")
            lines.append("            _nz_t = _reshaped.permute(_fwd_idx).reshape(*_bdims, _outer, _inner).contiguous().npu()")
            lines.append(f'            _nz_converted["{_nz_name}"] = _nz_t')
            lines.append("            _k1_p = _outer // 16")
            lines.append("            _n1_p = _inner // 16")
            lines.append("            if _mat2_transposed:")
            lines.append("                self._nz_storage_shape = torch.Size([_bdims[0] if _bdims else 1, _k1_p, _n1_p, 16, 16])")
            lines.append("            else:")
            lines.append("                self._nz_storage_shape = torch.Size([_bdims[0] if _bdims else 1, _n1_p, _k1_p, 16, 16])")
        lines.append("")
        lines.append("        # 转置时修正 output shape（CPU golden shape 不对 + NPU 用 NZ padding 后 n 校验 out）")
        lines.append("        if _self_transposed or _mat2_transposed:")
        lines.append('            _out_json = self._get_kwarg(input_data, "out")')
        lines.append("            if _out_json is not None:")
        lines.append("                _out_list = list(_out_json.shape)")
        lines.append("                _out_n_padded = ((_out_list[-1] + 15) // 16) * 16")
        lines.append("                self._correct_out_shape = torch.Size([_out_list[0], _out_list[-2], _out_n_padded])")
        lines.append("")

    lines.append("        # JSON inputs 参数顺序可能与签名不一致，统一通过 key 名称提取")
    lines.append("        # 取值顺序: kwargs key → args dict key → NULL")
    lines.append("        # 按签名中的完整参数顺序构建参数列表，保证 output tensor 在正确位置")
    lines.append("        for _p in self._SIG_ORDER:")
    lines.append("            _name = _p['name']")
    lines.append("            _kind = _p['kind']")
    lines.append("            _raw_type = _p['raw_type']")
    lines.append("            if _name in ('workspaceSize', 'executor'):")
    lines.append("                continue  # framework 参数，由 ATK 自动追加")
    lines.append("            if _kind == 'output':")
    lines.append("                if _output_idx < len(self.task_result.output_info_list):")
    lines.append("                    output_data = self.task_result.output_info_list[_output_idx]")
    if has_nz:
        lines.append("                    # 转置时覆盖 output shape 为 JSON 中的正确值")
        lines.append('                    if _name == "out" and hasattr(self, "_correct_out_shape"):')
        lines.append("                        output_data.shape = list(self._correct_out_shape)")
    lines.append("                    # 如果签名要求 aclTensorList* / aclScalarList* 类型的输出，需要从 output_info_list 中")
    lines.append("                    # 收集对应数量的 OutputData，打包成 list 传给 convert_output_data，")
    lines.append("                    # 这样才会走 create_x_list 分支创建正确的 List 类型（ATK update_output_info_list 会扁平化嵌套 list）")
    lines.append("                    if 'aclTensorList' in _raw_type or 'aclScalarList' in _raw_type:")
    lines.append("                        input_list_count = None")
    lines.append("                        for _ip in self._SIG_ORDER:")
    lines.append("                            if _ip['kind'] not in ('output', 'framework'):")
    lines.append("                                if 'aclTensorList' in _raw_type and 'aclTensorList' in _ip['raw_type']:")
    lines.append("                                    _ikw = input_data.kwargs.get(_ip['name'])")
    lines.append("                                    if _ikw and isinstance(_ikw, (list, tuple)) and len(_ikw) > 0:")
    lines.append("                                        input_list_count = len(_ikw)")
    lines.append("                                        break")
    lines.append("                                elif 'aclScalarList' in _raw_type and 'aclScalarList' in _ip['raw_type']:")
    lines.append("                                    _ikw = input_data.kwargs.get(_ip['name'])")
    lines.append("                                    if _ikw and isinstance(_ikw, (list, tuple)) and len(_ikw) > 0:")
    lines.append("                                        input_list_count = len(_ikw)")
    lines.append("                                        break")
    lines.append("                        if input_list_count is not None:")
    lines.append("                            _collected = []")
    lines.append("                            for _i in range(input_list_count):")
    lines.append("                                if _output_idx + _i < len(self.task_result.output_info_list):")
    lines.append("                                    _collected.append(self.task_result.output_info_list[_output_idx + _i])")
    lines.append("                            output = self.backend.convert_output_data(_collected, _output_idx)")
    lines.append("                            _output_idx += input_list_count  # 跳过已收集的 output_info 记录")
    lines.append("                        else:")
    lines.append("                            output = self.backend.convert_output_data(output_data, _output_idx)")
    lines.append("                    else:")
    lines.append("                        output = self.backend.convert_output_data(output_data, _output_idx)")
    lines.append("                    output_packages.extend(output)")
    lines.append("                    input_args.extend(output)")
    lines.append("                else:")
    lines.append("                    # 可选 output 参数超出 CPU 标杆返回的数量，填入 NULL 指针")
    lines.append("                    input_args.append(self._get_null_for_param(_name, _raw_type))")
    lines.append("                _output_idx += 1")
    lines.append("            else:")
    lines.append("                _val = self._get_kwarg(input_data, _name)")
    if has_always_null:
        lines.append("                if _name in self._ALWAYS_NULL_PARAMS:")
        lines.append("                    input_args.append(self._get_null_for_param(_name, _raw_type))")
    if has_nz:
        if has_always_null:
            lines.append("                elif _name in _nz_converted:")
        else:
            lines.append("                if _name in _nz_converted:")
        lines.append("                    _nz_shape = self.get_storage_shape(input_data, name=_name)")
        lines.append("                    _acl = nnopbase.create_acl_tensor(_nz_converted[_name], AclFormat.ACL_FORMAT_FRACTAL_NZ, _nz_shape)")
        lines.append("                    input_args.append(_acl)")
        lines.append("                elif _name in _self_converted:")
        lines.append("                    data = self.backend.convert_input_data(_self_converted[_name], name=_name)")
        lines.append("                    input_args.extend(data)")
        lines.append("                elif _val is not None:")
        lines.append("                    data = self.backend.convert_input_data(_val, name=_name)")
        lines.append("                    _INT_TYPES = {'int8_t', 'int32_t', 'int64_t', 'uint8_t', 'uint32_t', 'uint64_t', 'bool'}")
        lines.append("                    if _raw_type in _INT_TYPES and isinstance(data, list) and len(data) == 1:")
        lines.append("                        data = [self._ATTR_TYPE_MAP[_raw_type](int(getattr(data[0], 'value', data[0])))]")
        lines.append("                    input_args.extend(data)")
        lines.append("                else:")
        lines.append("                    input_args.append(self._get_null_for_param(_name, _raw_type))")
    else:
        if has_always_null:
            lines.append("                elif _val is not None:")
        else:
            lines.append("                if _val is not None:")
        lines.append("                    data = self.backend.convert_input_data(_val, name=_name)")
        lines.append("                    _INT_TYPES = {'int8_t', 'int32_t', 'int64_t', 'uint8_t', 'uint32_t', 'uint64_t', 'bool'}")
        lines.append("                    if _raw_type in _INT_TYPES and isinstance(data, list) and len(data) == 1:")
        lines.append("                        data = [self._ATTR_TYPE_MAP[_raw_type](int(getattr(data[0], 'value', data[0])))]")
        lines.append("                    input_args.extend(data)")
        lines.append("                else:")
        lines.append("                    input_args.append(self._get_null_for_param(_name, _raw_type))")
    lines.append("")
    lines.append("        return input_args, output_packages")
    lines.append("")
    lines.append("    def after_call(self, output_packages):")
    lines.append("        output = []")
    lines.append("        for output_pack in output_packages:")
    lines.append("            if isinstance(output_pack, AclTensorStruct):")
    lines.append("                output.append(self.acl_tensor_to_torch(output_pack))")
    lines.append("            elif isinstance(output_pack, AclTensorlistStruct):")
    lines.append("                output.append(self.acl_tensorlist_to_torch(output_pack))")
    lines.append("        return output")
    lines.append("")
    lines.append("    def get_cpp_func_signature_type(self):")
    lines.append(f'        return "{signature}"')
    lines.append("")
    lines.append("")
    lines.append("")

    # ========== Class 2+: BaseApi (CPU 后端 - 供大模型推导真实 PyTorch 调用) ==========
    # 统计签名中 output 参数的数量
    output_count = sum(1 for p in sig_params if p["kind"] == "output")
    # 收集 output 参数名
    output_param_names = [p["name"] for p in sig_params if p["kind"] == "output"]
    # 收集输入 tensor 参数（含 tensorList）
    input_tensor_params = [p for p in sig_params if p["kind"] in ("tensor", "tensorList")]
    # 收集 scalar 参数（含 scalarList）
    scalar_params = [p for p in sig_params if p["kind"] in ("scalar", "scalarList")]
    # 收集 attr 参数（不含 framework）
    attr_params = [p for p in sig_params if p["kind"] == "attr"]

    # 构建签名中所有非 framework/output 参数的注释行
    def _param_comment_line(p):
        kind_label = {"tensor": "input tensor", "scalar": "scalar", "scalarList": "scalar list", "tensorList": "tensor list", "attr": "attr", "output": "output tensor"}.get(p["kind"], p["kind"])
        return f"        # {p['name']:20s} ({p['raw_type']:22s})  # {kind_label}"

    for api_type in api_types:
        cpu_class_name = _to_class_name(api_type)
        lines.append(f'@register("{api_type}")')
        lines.append(f"class {cpu_class_name}(BaseApi):")
        lines.append(f'    """Auto-generated CPU reference class for {op_name}."""')
        lines.append("")
        lines.append(f'    _OP_NAME = "{op_name}"')
        lines.append(f'    _SIG_STR = """{signature}"""')
        # 记录签名中非 framework/output 的输入参数顺序（用于 args 模式取参）
        input_sig_names = [p["name"] for p in sig_params
                          if p["kind"] not in ("output", "framework")
                          and p["name"] not in FRAMEWORK_PARAMS]
        lines.append(f'    _INPUT_PARAM_NAMES = {input_sig_names}')
        lines.append("")
        lines.append("    def __call__(self, input_data: InputDataset, with_output: bool = False):")

        # —— 参数提取：从 args 按 case_config.inputs 的 name 构建映射 ——
        lines.append("        # Build name->value mapping from input_data.args using case_config.inputs names.")
        lines.append("        # ATK base_dataset.py appends each generated value to input_data.args in JSON inputs[] order.")
        lines.append("        # case_config.inputs[i].name gives the parameter name for args[i].")
        lines.append("        _param_map = {}")
        lines.append("        if input_data.args and hasattr(self, 'task_result') and self.task_result.case_config.inputs:")
        lines.append("            flat_configs = self.task_result.case_config.flatten_list(self.task_result.case_config.inputs)")
        lines.append("            for idx, conf in enumerate(flat_configs):")
        lines.append("                if idx < len(input_data.args) and conf.name:")
        lines.append("                    _param_map[conf.name] = input_data.args[idx]")
        lines.append("                    # Also index by lowercase for case-insensitive lookup")
        lines.append("                    _param_map[conf.name.lower()] = input_data.args[idx]")
        lines.append("        # Fallback: also index kwargs if present (for non-ACLNN paths)")
        lines.append("        if input_data.kwargs:")
        lines.append("            for k, v in input_data.kwargs.items():")
        lines.append("                if v is not None:")
        lines.append("                    _param_map[k] = v")
        lines.append("                    _param_map[k.lower()] = v")
        lines.append("")
        lines.append("        def _get_param(name, default=None):")
        lines.append("            v = _param_map.get(name)")
        lines.append("            if v is None:")
        lines.append("                v = _param_map.get(name.lower())")
        lines.append("            if v is not None:")
        lines.append("                return v")
        lines.append("            return default")
        lines.append("")

        # ——  tensor 类型验证辅助函数 ——
        lines.append("        # Tensor validation: ensures a param is actually a torch.Tensor")
        lines.append("        # ATK may sometimes generate non-tensor values (e.g. int) for params")
        lines.append("        # expected to be tensors, causing AttributeError during computation")
        lines.append("        def _get_tensor(name, default=None):")
        lines.append("            v = _get_param(name, default)")
        lines.append("            return v if isinstance(v, torch.Tensor) else default")
        lines.append("")

        # —— 在方法体内嵌入签名上下文，供大模型读取 ——
        lines.append(f"        # ACLNN operator: {op_name}")
        lines.append(f"        # C++ signature: {signature}")
        lines.append("")
        lines.append("        # --- Input parameters from signature ---")
        for p in sig_params:
            if p["name"] in ("workspaceSize", "executor"):
                continue
            lines.append(_param_comment_line(p))
        lines.append("")

        # —— TODO 标记：指导大模型从这里开始修改 ——
        lines.append("        # TODO: CPU_GOLDEN — Replace the dummy computation below with")
        lines.append("        # the real PyTorch CPU equivalent. Use the signature above to derive")
        lines.append("        # the correct torch.* call. Use _get_tensor(name) for tensors and")
        lines.append("        # _get_param(name, default) for scalars/attrs.")
        lines.append("        # _get_tensor validates isinstance(v, torch.Tensor) to prevent")
        lines.append("        # AttributeError when ATK generates wrong types for tensor params.")
        lines.append("")

        # —— 参数提取骨架（注释形式） ——
        # 判断可选参数：名字包含 Optional，或在 JSON inputs 中标记 required=false
        optional_names = {inp.get("name") for inp in _collect_all_inputs(cases)
                         if not inp.get("required", True)}
        if input_tensor_params:
            lines.append("        # Extract input tensors (use _get_tensor to validate type):")
            for p in input_tensor_params:
                if p["name"] in optional_names or "Optional" in p["name"]:
                    lines.append(f"        {p['name']} = _get_tensor(\"{p['name']}\", None)  # optional {p['raw_type']}")
                else:
                    lines.append(f"        {p['name']} = _get_tensor(\"{p['name']}\")  # {p['raw_type']}")
        if scalar_params:
            lines.append("        # Extract scalars:")
            for p in scalar_params:
                default = "1.0" if "alpha" in p["name"].lower() or "beta" in p["name"].lower() else "0"
                lines.append(f"        {p['name']} = _get_param(\"{p['name']}\", {default})")
        if attr_params:
            lines.append("        # Extract attrs:")
            for p in attr_params:
                lines.append(f"        # {p['name']} ({p['raw_type']}) = _get_param(\"{p['name']}\")")
        if input_tensor_params or scalar_params or attr_params:
            lines.append("")

        # —— 保留 dummy 逻辑作为 fallback ——
        if output_count == 0:
            # Inplace 算子：无 output 参数
            inplace_self = sig_params[0]["name"] if sig_params else "selfRef"
            lines.append("        # [FALLBACK] Inplace operator, no output tensor")
            lines.append("        tensors = [a for a in input_data.args if isinstance(a, torch.Tensor)]")
            lines.append("        if not tensors:")
            lines.append("            tensors = [v for v in input_data.kwargs.values() if isinstance(v, torch.Tensor)]")
            lines.append("        return next(iter(tensors)) if tensors else torch.tensor([])")
        else:
            lines.append("        # [FALLBACK] Dummy output — replace with real computation above")
            lines.append("        tensors = [a for a in input_data.args if isinstance(a, torch.Tensor)]")
            lines.append("        if not tensors:")
            lines.append("            tensors = [v for v in input_data.kwargs.values() if isinstance(v, torch.Tensor)]")
            lines.append("        dtype = next((v.dtype for v in tensors), torch.float32)")
            lines.append("        outputs = []")
            lines.append("        def _dummy_output(out_name):")
            lines.append("            candidates = [out_name]")
            lines.append('            for suffix in ("Out", "Optional"):')
            lines.append('                if out_name.endswith(suffix):')
            lines.append('                    stripped = out_name[:-len(suffix)]')
            lines.append('                    candidates.append(stripped)')
            lines.append('                    if stripped.startswith("grad"):')
            lines.append('                        candidates.append(stripped[4:])')
            lines.append('            if out_name.startswith("grad"):')
            lines.append('                candidates.append(out_name[4:])')
            lines.append("            for t in tensors:")
            lines.append("                tn = getattr(t, '_name', '')")
            lines.append("                for k in candidates:")
            lines.append("                    if tn and (tn == k or tn.lower() == k.lower()):")
            lines.append("                        return torch.ones(t.shape, dtype=dtype)")
            lines.append("            if tensors:")
            lines.append("                return torch.ones(tensors[0].shape, dtype=dtype)")
            lines.append("            return torch.ones([1], dtype=dtype)")
            for out_name in output_param_names:
                lines.append(f"        outputs.append(_dummy_output(\"{out_name}\"))")
            if output_count == 1:
                lines.append("        return outputs[0]")
            else:
                lines.append("        return outputs")

        lines.append("        # END_CPU_GOLDEN")
        lines.append("")
        lines.append("")

    return "\n".join(lines)


def missing_params_repr(missing_params: list[dict]) -> str:
    """生成缺失参数的 Python 表示"""
    items = []
    for p in missing_params:
        items.append(
            f'{{"name": "{p["name"]}", "raw_type": "{p["raw_type"]}", "kind": "{p["kind"]}"}}'
        )
    return "[" + ", ".join(items) + "]"


def _to_class_name(api_type: str) -> str:
    """
    将 api_type 字符串转换为 Python 类名。
    "pyaclnn_aclnn_addmv" -> "PyaclnnAclnnAddmv"
    "pyaclnn_aclnnAdaLayerNorm" -> "PyaclnnAclnnAdaLayerNorm"
    """
    # 替换下划线，然后驼峰
    parts = api_type.replace("-", "_").split("_")
    camel_parts = []
    for p in parts:
        if p.startswith("pyaclnn"):
            camel_parts.append("Pyaclnn")
        elif p.startswith("aclnn"):
            # 保留 aclnn 后的原始大小写
            camel_parts.append(p[0].upper() + p[1:])
        elif p:
            camel_parts.append(p.capitalize())
    name = "".join(camel_parts)
    # 如果包含大小写混合（如 AdaLayerNorm），直接用原始部分
    if not name:
        name = api_type.replace("-", "_").strip()
        name = name[0].upper() + name[1:] if name else "Api"
    return name


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ATK API 脚本生成器 — 从用例 JSON + aclnn.txt 生成可执行的 ATK py 文件"
    )
    parser.add_argument(
        "case_json",
        help="ATK 用例 JSON 文件路径（包含 问题 1：aclnn.txt 第 5 行签名有断行污染 — aclTen      sor *meanOutOptional 被分成了两行（aclTen 和 sor 中间有空格），load_signatures 读取时可能只读到了一行的一部分。 aclnn_name, api_type, inputs 等字段）",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 py 文件路径（默认: 与 JSON 同名，后缀改为 .py）",
    )
    parser.add_argument(
        "--signatures",
        default=os.path.join(os.path.dirname(__file__), "aclnn.txt"),
        help="aclnn.txt 签名表路径（默认: 脚本同目录下 aclnn.txt）",
    )
    args = parser.parse_args()

    # 加载签名表
    sigs = load_signatures(args.signatures)
    if not sigs:
        print(f"错误: 签名表为空或格式不正确: {args.signatures}", file=sys.stderr)
        sys.exit(1)

    # 加载用例 JSON
    with open(args.case_json, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 支持 JSON 为单个对象或数组
    if isinstance(cases, dict):
        cases = [cases]

    # 展开 inputs 中带 length 字段的 list 类型输入（tensors / scalars）为多个独立 input
    # 新格式: {"name": "x", "type": "tensors", "length": 2}
    # → 旧格式: [{"name": "x", "type": "tensors"}, {"name": "x", "type": "tensors"}]
    # 支持两种 inputs 结构:
    #   1) [[{dict}, {dict}, ...]] — 嵌套 list（多个输入组合）
    #   2) [{dict_with_length}] — 单层 list（一个输入组合，含 length 字段）
    # 同时清理 stale "length": null 字段（ATK 不需要）
    _LIST_TYPES = {"tensors", "scalars"}
    for case in cases:
        new_inputs = []
        for inp_group in case.get("inputs", []):
            if isinstance(inp_group, list):
                # 嵌套 list: 逐个展开
                expanded = []
                for inp in inp_group:
                    if isinstance(inp, dict) and inp.get("type") in _LIST_TYPES and "length" in inp:
                        length = inp.pop("length")
                        if length is None:
                            inp.pop("length", None)  # remove stale null
                            expanded.append(inp)
                        else:
                            for _ in range(int(length)):
                                expanded.append(copy.deepcopy(inp))
                    else:
                        expanded.append(inp)
                new_inputs.append(expanded)
            elif isinstance(inp_group, dict):
                # 单层 dict: 如果有 length，展开为多个 dict 组成的内层 list
                if inp_group.get("type") in _LIST_TYPES and "length" in inp_group:
                    length = inp_group.pop("length")
                    if length is None:
                        inp_group.pop("length", None)  # remove stale null
                        new_inputs.append([inp_group])
                    else:
                        new_inputs.append([copy.deepcopy(inp_group) for _ in range(int(length))])
                else:
                    new_inputs.append(inp_group)
            else:
                new_inputs.append(inp_group)
        case["inputs"] = new_inputs

    # 输出路径
    base = os.path.splitext(args.case_json)[0]

    # 写回展开后的 JSON，供 ATK 在服务器端读取
    # ATK 需要嵌套 list 格式 [[{entry1}, {entry2}, ...]]，不识别 length 简写
    # 文件名与原 JSON 相同，放在和生成的 py 文件同一目录
    expanded_json_path = base + "_expanded.json"
    with open(expanded_json_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"已生成: {expanded_json_path} (展开后的 JSON，用于 ATK 服务器端运行)")

    # 按算子分组 (同一个算子的所有用例合并生成一个 py 文件)
    op_cases: dict[tuple[str, str], list[dict]] = {}
    skipped_indices = []

    for i, case in enumerate(cases):
        aclnn_name = case.get("aclnn_name") or case.get("name", "")
        if not aclnn_name:
            print(f"警告: 第 {i} 条用例缺少 aclnn_name，跳过", file=sys.stderr)
            skipped_indices.append(i)
            continue

        # 在签名表中查找 —— 尝试多种匹配方式
        sig = sigs.get(aclnn_name)
        if not sig:
            prefixed = "aclnn" + aclnn_name
            sig = sigs.get(prefixed)
        if not sig:
            for k, v in sigs.items():
                if k.lower() == aclnn_name.lower() or k.lower() == prefixed.lower():
                    sig = v
                    break
        if not sig:
            print(
                f"警告: 在签名表中找不到 '{aclnn_name}' 的签名，跳过。"
                f"可用签名: {list(sigs.keys())}",
                file=sys.stderr,
            )
            skipped_indices.append(i)
            continue

        # 从签名提取算子名
        sig_match = re.search(r"aclnnStatus\s+(aclnn\w+)GetWorkspaceSize", sig)
        op_name = sig_match.group(1) if sig_match else aclnn_name

        key = (op_name, sig)
        if key not in op_cases:
            op_cases[key] = []
        op_cases[key].append(case)

    # 每个算子生成一个 py 文件
    op_count = len(op_cases)
    for idx, ((op_name, sig), op_case_list) in enumerate(op_cases.items()):
        code = generate_api_class_for_op(op_case_list, sig, op_name)

        if args.output and op_count == 1:
            out_path = args.output
        elif args.output:
            out_name = os.path.splitext(os.path.basename(args.output))[0]
            out_dir = os.path.dirname(args.output) or "."
            out_path = os.path.join(out_dir, f"{out_name}_{op_name}.py")
        else:
            out_path = f"{base}_{op_name}.py"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"已生成: {out_path} (算子: {op_name}, 包含 {len(op_case_list)} 条用例)")


if __name__ == "__main__":
    main()