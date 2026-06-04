"""Prompt template for parameter relation extraction."""

RELATION_EXTRACT_PROMPT = """\
你是一个参数关系提取专家。从下面的文档 section 内容中，提取所有参数之间的耦合关系。

## 关系类型定义
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
- "shape&value"：shape 与取值的耦合

## 提取规则
1. 只提取涉及两个或两个以上参数的关系，单参数自身约束不提取
2. 判断每段文本的"描述主体"——关系描述的是哪些参数之间的什么维度的耦合
3. 在参数表格中（无论是 HTML <table> 还是 Markdown |...| 格式），
   每一行的"说明"/"使用说明"列中如果提及了其他参数，
   且描述的是该行参数与其他参数之间的约束关系，则提取为一条关系。
   该行文本归属于该行第一列所标注的参数名，不要因为文本中提及了其他参数名就错误归属。
4. 在表格外部的段落中（如平台特定说明、约束说明），
   如果描述了多个参数之间的约束，也应提取
5. 如果关系存在平台前置条件（如"Atlas A2/A3 下"），记录到 precondition 字段
6. 如果没有平台前置条件，precondition 填 "无"
7. params 字段列出关系涉及的所有参数名
8. param_optional 字段标注每个参数是否可选（从文档中的"可选参数"等描述判断）
9. source_citation 字段填写原文中描述该关系的原始文本
10. description 字段用自然语言简洁描述该关系

## 输出格式
严格按以下 JSON 格式返回，不要添加任何其他文字：
[
  {{
    "relation_type": "shape",
    "precondition": "无",
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
