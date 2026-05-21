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
