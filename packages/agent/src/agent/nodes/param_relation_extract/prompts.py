"""Prompt template for parameter relation extraction."""

RELATION_TYPE_DEFINITIONS = """\
每条关系的 relation_type 必须是以下值之一：
- "dtype"：仅涉及数据类型的约束或依赖
- "shape"：仅涉及维度/shape 的约束或依赖
- "dformat"：仅涉及数据格式的约束或依赖
- "value"：仅涉及参数取值的约束或依赖
- "dtype&shape"：同时涉及数据类型和 shape
- "dtype&dformat"：同时涉及数据类型和数据格式
- "dformat&shape"：同时涉及数据格式和 shape
- "dtype&dformat&shape"：三者同时涉及
- "presence"：存在性依赖（一个参数是否存在/为空取决于另一个参数的状态）
- "shape&value"：shape 与取值的耦合"""

RELATION_EXTRACT_PROMPT = """\
你是一个参数关系提取专家。从下面的文档 section 内容中，提取所有参数之间的耦合关系。

## 关系类型定义
{relation_types}

{implicit_params_context}
## 提取规则
1. 只提取涉及两个或两个以上参数的关系，单参数自身约束不提取
2. 判断每段文本的"描述主体"——关系描述的是哪些参数之间的什么维度的耦合
3. 在参数表格中（无论是 HTML <table> 还是 Markdown |...| 格式），
   每一行的"说明"/"使用说明"列中如果提及了其他参数，
   且描述的是该行参数与其他参数之间的约束关系，则提取为一条关系。
   该行文本归属于该行第一列所标注的参数名，不要因为文本中提及了其他参数名就错误归属。
4. 在表格外部的段落中（如平台特定说明、约束说明），
   如果描述了多个参数之间的约束，也应提取
5. platform 字段用于填写该关系适用的平台名称：
   - 如果关系适用于所有平台（文档中未指定特定平台），platform 填空字符串 ""
   - 如果关系仅适用于特定平台，填写平台名称
   - 如果关系适用于多个特定平台，用"、"分隔各平台名称
   - platform 只能填写平台名称，禁止填写非平台信息（如数据格式、参数值等约束条件）
6. 标准平台名称（必须严格使用以下格式）：
   - "Atlas 训练系列产品"
   - "Atlas 推理系列产品"
   - "Atlas A2 训练系列产品/Atlas A2 推理系列产品"
   - "Atlas A3 训练系列产品/Atlas A3 推理系列产品"
   - "Atlas 200I/500 A2 推理产品"
   - "Atlas 300I 推理产品"
   - "Atlas 300I Duo 推理产品"
   - "Atlas 300V 视频解析产品"
   - "Atlas 500 A2 智能小站"
   - "Atlas 800 推理服务器 A2"
   - "Atlas 800 训练服务器"
   - "Atlas 800I A2 推理服务器"
7. params 字段列出关系涉及的所有参数名
8. param_optional 字段标注每个参数是否可选（从文档中的"可选参数"等描述判断）
9. source_citation 字段填写原文中描述该关系的原始文本
10. description 字段用自然语言简洁描述该关系

## 输出格式
严格按以下 JSON 格式返回，不要添加任何其他文字：
[
  {{
    "relation_type": "shape",
    "platform": "",
    "description": "scale 的 shape 依赖 x 的 shape 和 axis：当 scale 为 1 维时，...",
    "params": ["x", "scale", "axis"],
    "param_optional": {{"x": false, "scale": false, "axis": false}},
    "source_citation": "`scale`支持1维张量或多维张量，shape与输入`x`和属性`axis`有关..."
  }}
]

如果没有提取到任何关系，返回空数组 []

## Section 内容：
{section_content}
"""

# ---------------------------------------------------------------------------
# Implicit (non-operator) parameters context (injected into prompt)
# ---------------------------------------------------------------------------

IMPLICIT_PARAMS_CONTEXT = """\
## 非算子参数（命名维度变量）
以下是从 shape 描述中提取的非算子参数（命名维度变量），
它们虽然不是函数签名中的参数，但在 shape 约束中作为有意义的维度变量出现。
请将它们视为正式参数，在 params 列表中包含，并在表达式中直接使用其名称。

非算子参数列表：
{mapping_list}

注意：
1. params 列表中应包含所有涉及的参数，包括算子参数和非算子参数
2. 表达式中使用命名变量名（如 BS.range_value、H.range_value）
3. 不要使用 tensor.shape[i] 替代命名变量
4. 对于常量维度（如 k0=16），在表达式中直接使用数值
5. description/source_citation 中可保留原始描述

"""


def format_implicit_params_context(
    implicit_params: list[dict],
    platform_constants: list[dict] | None = None,
) -> str:
    """Build the implicit parameters context string for prompt injection.

    Returns empty string when no implicit params exist, so the prompt
    section is omitted entirely.

    Enhanced with external constant section when platform_constants
    or external constant mappings are present.
    """
    if not implicit_params:
        return ""

    # Handle old-style implicit_params data (backward compat)
    if implicit_params and "param_name" in implicit_params[0] and "var_name" not in implicit_params[0]:
        lines = []
        for ip in implicit_params:
            name = ip.get("param_name", "")
            ptype = ip.get("param_type", "int64_t")
            lines.append(f"- {name}（{ptype}）：隐式维度变量")
        old_context = """\
## 隐式维度变量（非函数签名参数，但在 shape 描述中作为命名维度使用）
以下标识符虽然不是函数签名中的参数，但它们是重要的维度变量，\
在 shape 描述中以命名形式出现。请将它们视为正式参数，\
并在涉及它们的约束关系中将其列入 params 列表：
{implicit_params_list}

"""
        return old_context.format(
            implicit_params_list="\n".join(lines)
        )

    # New-style: implicit params data (dicts with var_name/tensor_param)
    from collections import defaultdict

    var_groups: dict[str, list[str]] = defaultdict(list)
    constants: dict[str, int] = {}
    ext_consts: list[dict] = []
    quant_entry: dict | None = None

    for m in implicit_params:
        var = m["var_name"]
        if m.get("is_quantization_type"):
            quant_entry = m
            continue
        if m.get("is_external_constant"):
            ext_consts.append(m)
            continue
        if m.get("is_constant"):
            constants[var] = m.get("constant_value", 0)
        else:
            ref = f'{m["tensor_param"]}.shape[{m["dim_index"]}]'
            if ref not in var_groups[var]:
                var_groups[var].append(ref)

    lines: list[str] = []
    for var in sorted(var_groups):
        refs = var_groups[var]
        lines.append(f"- {var}（非算子参数）：对应 {', '.join(refs)}")
    for var in sorted(constants):
        val = constants[var]
        tensor_refs = [
            f'{m["tensor_param"]}.shape[{m["dim_index"]}]'
            for m in implicit_params
            if m["var_name"] == var
        ]
        ref_str = f"，对应 {', '.join(tensor_refs)}" if tensor_refs else ""
        lines.append(f"- {var} = {val}（常量，直接使用数值{ref_str}）")

    # Quantization type section (default char-typed enum implicit param)
    if quant_entry:
        modes = quant_entry.get("allowed_range_value", [])
        modes_str = "、".join(modes) if modes else "无（文档未明确）"
        lines.append("")
        lines.append("## 量化粒度参数（char 类型枚举）")
        lines.append(
            "quantization_type 是非算子参数（char 类型枚举），"
            "表示量化粒度模式。"
        )
        lines.append(
            f"其允许取值（allowed_range_value）为：{modes_str}。"
        )
        lines.append(
            "在涉及量化场景的约束表达式中，以 "
            "quantization_type.range_value == \"per-channel\" "
            "等形式引用，并将其列入 params 列表。"
        )

    # External constants section
    if ext_consts or platform_constants:
        lines.append("")
        lines.append(
            "## 平台外部常量（非函数签名参数，"
            "平台级常量）"
        )
        lines.append(
            "以下变量是平台级外部常量，在约束表达式中"
            "直接以 名称.range_value 形式引用。"
        )
        lines.append("在 params 列表中不要列入这些常量。")
        lines.append("")

        for ec in ext_consts:
            var = ec["var_name"]
            refs = ec.get("referenced_in", [])
            ref_str = (
                f"（出现在 {', '.join(refs)} 的 shape 表达式中）"
                if refs else ""
            )
            lines.append(f"- {var} → 平台外部常量{ref_str}")

        if platform_constants:
            lines.append("")
            lines.append("平台取值约束：")
            for pc in platform_constants:
                for pv in pc.get("platform_values", []):
                    lines.append(
                        f"- {pc['const_name']} 在 "
                        f"{pv['platform']} 上取值 "
                        f"{pv['values']}"
                    )

    return IMPLICIT_PARAMS_CONTEXT.format(
        mapping_list="\n".join(lines)
    )
