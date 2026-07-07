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
import jinja2

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

    使用 Jinja2 模板 (aclnn_api_template.py.j2) 渲染输出。
    特殊算子 (aclnnCalculateMatmulWeightSize / V2) 直接返回预定义模板。
    """
    _SPECIAL_TEMPLATES = {
        "aclnnCalculateMatmulWeightSize": "aclnnCalculateMatmulWeightSize.py.tpl",
        "aclnnCalculateMatmulWeightSizeV2": "aclnnCalculateMatmulWeightSizeV2.py.tpl",
    }
    if op_name in _SPECIAL_TEMPLATES:
        tpl_path = os.path.join(os.path.dirname(__file__), _SPECIAL_TEMPLATES[op_name])
        with open(tpl_path, "r", encoding="utf-8") as f:
            return f.read()

    sig_params = parse_cpp_signature(signature)

    # 从 JSON outputs 字段读取输出参数名，覆盖签名解析结果
    output_names = set()
    for _case in cases:
        _outputs = _case.get("outputs")
        if _outputs:
            if isinstance(_outputs, list):
                output_names.update(_outputs)
            elif isinstance(_outputs, str):
                output_names.update(n.strip() for n in _outputs.split(",") if n.strip())
    if output_names:
        for _p in sig_params:
            if _p["name"] in output_names:
                _p["kind"] = "output"

    api_types, aclnn_api_type = _collect_api_types(cases)
    if not aclnn_api_type:
        aclnn_api_type = "pyaclnn_aclnn_" + op_name.lower()
    aclnn_class_name = _to_class_name(op_name)

    # ---- CPU 类预计算 ----
    output_count = sum(1 for p in sig_params if p["kind"] == "output")
    output_param_names = [p["name"] for p in sig_params if p["kind"] == "output"]
    input_tensor_params = [p for p in sig_params if p["kind"] in ("tensor", "tensorList")]
    scalar_params = [p for p in sig_params if p["kind"] in ("scalar", "scalarList")]
    attr_params = [p for p in sig_params if p["kind"] == "attr"]

    input_sig_names = [p["name"] for p in sig_params
                       if p["kind"] not in ("output", "framework")
                       and p["name"] not in FRAMEWORK_PARAMS]

    optional_names = {inp.get("name") for inp in _collect_all_inputs(cases)
                      if not inp.get("required", True)}

    # 参数注释行
    _kind_labels = {"tensor": "input tensor", "scalar": "scalar", "scalarList": "scalar list",
                    "tensorList": "tensor list", "attr": "attr", "output": "output tensor"}
    param_comments = []
    for p in sig_params:
        if p["name"] in ("workspaceSize", "executor"):
            continue
        kind_label = _kind_labels.get(p["kind"], p["kind"])
        param_comments.append(f"        # {p['name']:20s} ({p['raw_type']:22s})  # {kind_label}")

    # tensor 提取行
    tensor_lines = []
    for p in input_tensor_params:
        if p["name"] in optional_names or "Optional" in p["name"]:
            tensor_lines.append(f'        {p["name"]} = _get_tensor("{p["name"]}", None)  # optional {p["raw_type"]}')
        else:
            tensor_lines.append(f'        {p["name"]} = _get_tensor("{p["name"]}")  # {p["raw_type"]}')

    # scalar 提取行
    scalar_lines = []
    for p in scalar_params:
        default = "1.0" if "alpha" in p["name"].lower() or "beta" in p["name"].lower() else "0"
        scalar_lines.append(f'        {p["name"]} = _get_param("{p["name"]}", {default})')

    # attr 提取行
    attr_lines = []
    for p in attr_params:
        attr_lines.append(f'        # {p["name"]} ({p["raw_type"]}) = _get_param("{p["name"]}")')

    # output append 行
    output_append_lines = [f'        outputs.append(_dummy_output("{out_name}"))' for out_name in output_param_names]

    cpu_classes = [(api_type, _to_class_name(api_type)) for api_type in api_types]

    # ---- 渲染模板 ----
    template_path = os.path.join(os.path.dirname(__file__), "aclnn_api_template.py.j2")
    with open(template_path, "r", encoding="utf-8") as f:
        template_source = f.read()
    template = jinja2.Template(template_source, trim_blocks=True, lstrip_blocks=True)

    return template.render(
        aclnn_api_type=aclnn_api_type,
        class_name=aclnn_class_name,
        signature=signature,
        op_name=op_name,
        cpu_classes=cpu_classes,
        input_sig_names=input_sig_names,
        param_comments=param_comments,
        tensor_lines=tensor_lines,
        scalar_lines=scalar_lines,
        attr_lines=attr_lines,
        output_count=output_count,
        output_append_lines=output_append_lines,
    )

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

    # 展开 inputs 中带 length 字段的 list 类型输入（tensors / scalars / attrs）为多个独立 input
    # 新格式: {"name": "x", "type": "tensors", "length": 2}
    # → 旧格式: [{"name": "x", "type": "tensors"}, {"name": "x", "type": "tensors"}]
    # 支持两种 inputs 结构:
    #   1) [[{dict}, {dict}, ...]] — 嵌套 list（多个输入组合）
    #   2) [{dict_with_length}] — 单层 list（一个输入组合，含 length 字段）
    # 同时清理 stale "length": null 字段（ATK 不需要）
    # attrs 类型展开规则（type 改为 attr，删除 length）:
    #   1. range_values 是列表且 len != length → 复制 length 次，每个保持原始 range_values
    #   2. range_values 是列表且 len == length → 复制 length 次，每个取 range_values[i]
    #   3. length 为 None/0 → 单个条目，range_values 设为空列表 []
    def _expand_attrs_input(inp):
        rv = inp.get("range_values")
        length = inp.get("length")
        base = {k: v for k, v in inp.items() if k != "length"}
        # type 保持为 attrs
        if length is None or int(length) == 0:
            base["range_values"] = []
            return [base]
        length = int(length)
        if isinstance(rv, list):
            rv_list = rv
        elif rv is not None:
            rv_list = [rv]
        else:
            rv_list = []
        if len(rv_list) == length:
            result = []
            for i in range(length):
                item = copy.deepcopy(base)
                item["range_values"] = rv_list[i]
                result.append(item)
            return result
        return [copy.deepcopy(base) for _ in range(length)]

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
                    elif isinstance(inp, dict) and inp.get("type") == "attrs":
                        if "length" in inp:
                            expanded.extend(_expand_attrs_input(copy.deepcopy(inp)))
                        else:
                            expanded.append(inp)
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
                elif inp_group.get("type") == "attrs":
                    if "length" in inp_group:
                        new_inputs.append(_expand_attrs_input(copy.deepcopy(inp_group)))
                    else:
                        new_inputs.append(inp_group)
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
    # 一段式算子特殊处理: aclnnCalculateMatmulWeightSize / V2 的 aclnn_name 改为 Ad
    _SPECIAL_ONE_STAGE_OPS = {"aclnnCalculateMatmulWeightSize", "aclnnCalculateMatmulWeightSizeV2"}
    expanded_cases = copy.deepcopy(cases)
    for case in expanded_cases:
        case_name = case.get("aclnn_name", "") or case.get("name", "")
        if case_name in _SPECIAL_ONE_STAGE_OPS:
            case["aclnn_name"] = "Add"
    expanded_json_path = base + "_expanded.json"
    with open(expanded_json_path, "w", encoding="utf-8") as f:
        json.dump(expanded_cases, f, ensure_ascii=False, indent=2)
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