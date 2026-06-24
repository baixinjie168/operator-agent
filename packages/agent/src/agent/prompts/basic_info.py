"""Prompts for basic document info extraction: function signatures, product support, function explanation."""

FUNCTION_SIGNATURE_EXTRACT_PROMPT = """\
你是一个 C 函数签名提取专家。从下面的函数原型内容中，提取所有函数的完整签名信息。

要求：
1. 每个函数对应一个对象
2. function_name: 函数名
3. return_type: 返回值类型
4. parameters: 参数列表，每个参数包含 name（参数名）和 type（C 基础类型名，去掉 const、指针符号 * 等修饰符）
5. full_signature: 完整的函数签名字符串，格式为 "返回值类型 函数名(参数1类型 *参数1名, 参数2类型 *参数2名, ...)"
   - 保留原始代码中的 const、指针 * 等修饰符
   - 参数之间用逗号和空格分隔
6. 严格按以下 JSON 格式返回，不要添加任何其他文字：

[
  {{
    "function_name": "函数名",
    "return_type": "返回值类型",
    "parameters": [
      {{"name": "x", "type": "aclTensor"}},
      {{"name": "workspaceSize", "type": "uint64_t"}}
    ],
    "full_signature": "返回值类型 函数名(参数类型 *参数名, ...)"
  }}
]

函数原型内容：
{content}"""


PRODUCT_SUPPORT_EXTRACT_PROMPT = """\
你是一个产品支持信息提取专家。从下面的 Markdown 表格内容中，提取每行产品及其是否支持。

要求：
1. 去除所有 HTML/XML 标签（如 <term>、</term>），只保留纯文本产品名
2. "√"、"✔"、"✓" 表示支持，"×"、"✗"、"-" 表示不支持
3. 忽略表头行和分隔行（如 |---|---|）
4. 严格按以下 JSON 格式返回，不要添加任何其他文字：

[
  {{"product": "Atlas A3 训练系列产品/Atlas A3 推理系列产品", "support": true}},
  {{"product": "Atlas 200I/500 A2 推理产品", "support": false}}
]

表格内容：
{content}"""


FUNCTION_EXPLANATION_EXTRACT_PROMPT = """\
你是一个算子功能说明提取专家。从下面的功能说明章节内容中，
为算子 "{operator_name}" 提取结构化的功能说明摘要。

要求：
1. description: 用 1-3 句话概括算子的核心功能，保持简洁准确
2. formula: 提取计算公式，使用 LaTeX 或纯文本数学表达式；无公式则留空
3. key_points: 提取关键特性列表（适用场景、与其他算子区别、特殊行为等），
   最多 5 条
4. source_text: 保留原始功能说明文本（去除 Markdown 标记后的纯文本）
5. 不要添加原文中不存在的信息
6. 去除 HTML/XML 标签（如 <term>），只保留纯文本

严格按以下 JSON 格式返回，不要添加任何其他文字：
{{{{
  "description": "算子功能的简要描述",
  "formula": "计算公式（LaTeX 或空字符串）",
  "key_points": ["关键特性1", "关键特性2"],
  "source_text": "原始功能说明文本"
}}}}

功能说明内容：
{content}"""
