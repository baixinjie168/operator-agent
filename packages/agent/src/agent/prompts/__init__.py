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


SRC_CONTENT_EXTRACT_PROMPT = """\
你是一个参数原文提取专家。从下面的文档内容中，找出与参数 "{param_name}" 直接相关的所有原始句子或段落。

要求：
1. 只提取文档中的原话，不要添加任何自己的理解、总结或改写
2. 每个条目必须是一条完整的原文句子或段落，不要截断也不要合并
3. 如果文档中有包含该参数的表格行，提取该行的原始 Markdown 表格行文本作为一个条目
4. 条目按文档中出现顺序排列
5. 如果文档中没有任何与该参数直接相关的原文，只输出一行"（无）"

主题归属规则（非常重要）：
6. 判断每段文本的"描述主体"是谁——即这段话主要是在描述哪个参数的属性、约束或行为。
   - 在参数表格中，每一行（<tr>）的文本归属于该行第一个单元格所标注的参数，
     不要因为文本中提及了其他参数名就把文本归到那个参数。
   - 例如：在参数 `x` 的使用说明中写道"如果`dstType`为3，shape的最后一维需要能被8整除"，
     这段话的描述主体是 `x`（描述的是 `x` 的 shape 约束），
     `dstType` 只是作为条件被引用，因此这段文本只属于 `x`，不属于 `dstType`。
7. 仅当文本的描述主体是 "{param_name}" 时才提取。
   如果 "{param_name}" 仅作为其他参数描述中的条件、引用或对比出现，不要提取该文本。
8. 在表格外部的段落中（如产品特定说明），如果句子的主语或主题是 "{param_name}"
  （如"入参`{param_name}`支持取值..."），则提取；
   如果 "{param_name}" 只是在描述其他参数时被顺带提及，则不提取。

严格按以下编号列表格式输出，不要添加任何前缀、解释或其他格式：
1. [从文档中直接引用的原文句子或段落]
2. [从文档中直接引用的原文句子或段落]
...

文档内容：
{content}
"""


PARAM_DESC_EXTRACT_PROMPT = """\
你是一个参数描述提取专家。从下面的原文引用内容中，为参数 "{param_name}" 整理出结构化的描述表格。

要求：
1. 只使用下面提供的«原文引用»内容，不要添加自己的理解或补充信息
2. 尽可能保留原文的措辞和标点
3. 严格返回纯文本，不要添加任何前缀、解释或 JSON 格式

输出格式：
**必须以 Markdown 表格形式输出**，表格结构如下（两列：属性、值）：

| 属性 | 值 |
|------|-----|
| 参数名 | [从原文中提取的内容] |
| 输入/输出 | [从原文中提取的内容] |
| 描述 | [从原文中提取的内容] |
| 使用说明 | [从原文中提取的内容，多行用<br>分隔] |
| 数据类型 | [从原文中提取的内容] |
| 数据格式 | [从原文中提取的内容] |
| 维度(shape) | [从原文中提取的内容] |
| 非连续Tensor | [从原文中提取的内容] |
| 其他 | [从原文中提取的内容] |

如果某类信息在原文引用中不存在，对应单元格填"（无）"
5. "非连续Tensor"行的值必须使用 √ 表示支持，× 表示不支持，不要使用"支持"/"不支持"等文字
只输出上述表格，不要输出表格之外的任何内容

原文引用：
{src_content}
"""


DTYPE_EXTRACT_PROMPT = """\
你是一个参数数据类型提取专家。从下面参数的 Markdown 表格描述中，提取其数据类型。

规则：
1. 首先看"数据类型"行，如果有具体值（不是"（无）"、"无"、空），则直接提取
2. 如果"数据类型"行是"（无）"，再看"描述"和"使用说明"行中是否有数据类型相关信息
3. 常见数据类型示例：INT8, INT16, INT32, INT64, FLOAT16, FLOAT32, FLOAT64,
   BOOL, UINT8, STRING, TENSOR, SCALAR, LIST, BFLOAT16, COMPLEX64, COMPLEX128
4. 提取的数据类型值必须统一为大写，如"INT8"、"FLOAT32"、"TENSOR"
5. 如果完全没有数据类型相关信息，dtype设为空字符串

严格按以下JSON格式返回，不要添加任何其他文字：
{{"param_name": "{param_name}", "dtype": "提取的大写数据类型或空字符串"}}

参数描述：
{params_text}
"""


OPTIONAL_EXTRACT_PROMPT = """\
你是一个参数可选性判断专家。从下面参数的 Markdown 表格描述中，判断该参数是否可选。

关键词清单（描述中出现以下任一表达即视为可选）:
- 中文: 可选、可空、可为空、可不传、不传、非必选、缺省值、默认值、默认
- 英文: Optional、optional、default、None、null、nullable
- 其他: 可以为 nullptr、可为 nullptr、空指针、未指定时

重要规则:
- 参数名为 "{param_name}"
- 参数名包含 Optional: {name_has_optional}
- 如果参数名包含 Optional，请忽略参数名，只看 Markdown 表格「值」列中的内容
- 只关注表格中「值」列的内容，不要把「属性」列（描述、数据类型等）当作判断依据
- 如果描述内容中确实出现了上述关键词，返回 true，否则返回 false
- 请注意：如果描述中出现"必选"、"必须"、"不可为空"等字眼，则不是可选，返回 false

严格按以下JSON格式返回，不要添加任何其他文字:
{{"param_name": "{param_name}", "is_optional": true}}

参数描述:
{params_text}
"""


DFORMAT_EXTRACT_PROMPT = """\
你是一个参数数据格式提取专家。从下面参数的 Markdown 表格描述中，提取其数据格式（Data Format）。

规则：
1. 首先看"数据格式"行，如果有具体值（不是"（无）"、"无"、空），则直接提取
2. 如果"数据格式"行是"（无）"，再看"描述"和"使用说明"行中是否有数据格式相关信息
3. 常见数据格式示例：ND, NC, NCL, NCHW, NCDHW, NHWC, NWC, NC1HWC0, NC1HWC0_C4,
   FRACTAL_Z, FRACTAL_NZ, FRACTAL_ZZ, FRACTAL_ZN_LSTM, NDHWC, NDC1HWC0
4. 提取的数据格式值必须统一为大写
5. 如果完全没有数据格式相关信息，dformat设为空字符串

严格按以下JSON格式返回，不要添加任何其他文字：
{{"param_name": "{param_name}", "dformat": "提取的数据格式或空字符串"}}

参数描述：
{params_text}
"""


SHAPE_EXTRACT_PROMPT = """\
你是一个参数shape提取专家。从下面参数的 Markdown 表格描述中，提取其维度(shape)信息。

规则：
1. 首先看"维度(shape)"行，如果有具体值（不是"（无）"、"无"、空），且该值不依赖其他参数的取值，则直接提取
2. 如果"维度(shape)"行是"（无）"，再看"使用说明"和"描述"行中是否有shape相关信息
3. 关键判断：描述中如果出现了"当..."、"如果..."、"若..."、"取决于"等条件从句
   （意为shape受其他参数取值影响），说明有前置条件，应忽略，shape设为空字符串
4. 只有无条件、直接给出的shape值才提取
5. 提取的shape值应简洁，保留原文关键信息，如"1-8"、"与输入相同"、"(N, C, H, W)"、"2D"
6. 如果完全没有shape相关信息，shape设为空字符串

严格按以下JSON格式返回，不要添加任何其他文字：
{{"param_name": "{param_name}", "shape": "提取的shape值或空字符串"}}

参数描述：
{params_text}
"""


FUNCTION_SIGNATURE_EXTRACT_PROMPT = """\
你是一个 C 函数签名提取专家。从下面的函数原型内容中，提取所有函数的完整签名信息。

要求：
1. 每个函数对应一个对象
2. function_name: 函数名
3. return_type: 返回值类型
4. parameters: 参数列表，每个参数包含 name（参数名）和 type（C 类型，去掉指针符号 *）
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


ARRAY_LENGTH_EXTRACT_PROMPT = """\
你是一个数组长度提取专家。从下面参数的 Markdown 表格描述中，提取该数组参数的长度/大小约束信息。

提取规则:
1. 只关注描述中关于数组长度、大小、元素个数的约束
2. 常见表达: "列表长度不超过N"、"最大长度为N"、"size大小为N"、"维度为N"、
   "支持的最大TensorList长度为N"、"长度范围[M, N]" 等
3. 如果描述中明确提到了长度约束，提取该约束的完整描述文本
4. 如果没有提到任何长度相关信息，返回空字符串

严格按以下JSON格式返回，不要添加任何其他文字:
{{"array_length": "提取到的长度描述或空字符串"}}

参数描述:
{params_text}
"""


ALLOWED_RANGE_EXTRACT_PROMPT = """\
你是一个参数取值范围提取专家。从下面的文档章节内容中，提取参数 "{param_name}"（类型: {param_type}）的取值范围约束。

提取规则:
1. 只关注与参数 "{param_name}" 直接相关的取值范围、取值约束、枚举值限制
2. 参数名在文档中可能以缩写或别名出现，需要根据上下文判断
3. 如果约束有平台限制（如"Atlas A2 下"、"Atlas A3"等），需要标注 platform 字段
4. 如果约束没有平台限制（通用约束），platform 设为空字符串 ""
5. 如果完全没有该参数的取值范围信息，返回空数组 []
6. allowed_range_value 保留原文的关键描述，简洁但完整
7. 不要提取 shape、数据类型、数据格式等其他约束，只提取取值范围/取值约束

严格按以下 JSON 数组格式返回，不要添加任何其他文字:
[
  {{"platform": "平台名称或空字符串", "allowed_range_value": "取值范围描述"}}
]

无约束时返回: []

文档章节内容:
{sections_text}
"""


RETURN_CODE_EXTRACT_PROMPT = """\
你是一个返回值错误码提取专家。从下面的文档章节内容中，提取所有错误码信息。

提取规则：
1. 提取每个错误码的 return_value（返回码名称）、error_code（错误码数值）和 descriptions（描述列表）
2. 同一个 (return_value, error_code) 组合如果有多条描述，合并到 descriptions 数组中
3. descriptions 中的每条描述应保持原文措辞，去掉开头的编号（如"1."、"2."）和前导空格
4. error_code 必须是整数类型（如 161001，不是字符串）
5. 如果章节中没有错误码表格或错误码信息，返回空数组 []
6. 支持三种格式：
   - Markdown表格：| 返回码 | 错误码 | 描述 |
   - 纯文本：返回161001 (ACLNN_ERR_PARAM_NULLPTR)：描述
   - HTML表格（含rowspan合并单元格）

严格按以下 JSON 数组格式返回，不要添加任何其他文字：
[
  {{
    "return_value": "ACLNN_ERR_PARAM_NULLPTR",
    "error_code": 161001,
    "descriptions": ["描述1", "描述2"]
  }}
]

无错误码信息时返回: []

章节内容：
{section_content}
"""


DETERMINISM_EXTRACT_PROMPT = """\
你是一个确定性计算提取专家。从下面的约束说明章节中，提取函数的确定性计算信息。

提取规则：
1. 查找"确定性计算"或"确定性说明"相关描述
2. 判断每个描述的确定性：
   - "确定性实现"、"默认确定性实现" → value = true
   - "非确定性实现"、"默认非确定性实现" → value = false
3. 提取平台信息：
   - 如果描述中明确指定了平台（如 `<term>Atlas xxx</term>`、`Atlas xxx`），提取平台名称到 product 字段
   - 如果有多个平台用"、"或"/"分隔，将它们作为一个完整的 product 字符串
   - 如果描述中未指定平台，product 设为空字符串 ""
4. src_text 保留原文完整描述（含条件说明，如有）
5. 如果没有找到确定性相关信息，返回空数组 []

严格按以下 JSON 数组格式返回，不要添加任何其他文字：
[
  {{
    "product": "Atlas A2 训练系列产品/Atlas A2 推理系列产品",
    "value": true,
    "src_text": "aclnnAddLayerNorm默认确定性实现。"
  }}
]

未指定平台时 product 为空字符串：
[
  {{
    "product": "",
    "value": true,
    "src_text": "aclnnAdaLayerNorm默认确定性实现。"
  }}
]

无确定性相关信息时返回: []

已知支持的平台列表（仅用于参考，不要编造平台）：
{platform_list}

章节内容：
{section_content}
"""


DTYPE_COMBO_TABLE_PROMPT = """\
你是一个 CANN 算子数据类型组合提取专家。从下面的数据类型组合表中，提取每一行的参数 dtype 组合。

规则：
1. 表头中每个列名映射到参数名（去掉"数据类型"后缀和反引号）
2. 每一行数据是一个完整的合法组合
3. 单元格含 "/" 时（如 "UINT64/INT64"），保留为字符串值，不要拆分
4. 单元格含 "null" 时，该参数在组合中省略（不出现在对象中）
5. 仅输出 JSON，不要添加任何其他文字
6. 如果有多个平台各自有一张子表，为每个平台分别输出一个对象

参数列表（用于参考参数名）：
{params_text}

组合表内容：
{table_text}

严格按以下 JSON 格式返回：
[
  {{
    "platform": "平台名称",
    "rows": [
      {{"x1": "FLOAT32", "x2": "FLOAT16", "gamma": "FLOAT32"}},
      {{"x1": "BFLOAT16", "x2": "BFLOAT16", "gamma": "BFLOAT16"}}
    ]
  }}
]

无平台限定时 platform 设为 "通用"。
"""


DTYPE_CONSTRAINT_TEXT_PROMPT = """\
你是一个 CANN 算子数据类型约束提取专家。从约束说明中提取每个参数的 dtype 约束信息。

规则：
1. 正向列举（"x1、x2 支持 FLOAT32、FLOAT16"）→ mode 设为 "positive"
2. 负向排除（"input 不支持 BFLOAT16"）→ mode 设为 "negative"
3. 识别平台限定（如 "Atlas A2 训练系列产品/Atlas A2 推理系列产品"）
4. 无平台限定时 platform 设为 "通用"
5. "、"连接的参数列表拆为多条记录（每个参数一条）
6. 如果约束说明中没有任何数据类型相关信息，返回空数组 []
7. 只提取与「数据类型」相关的描述，忽略 shape、数据格式等其他约束

参数列表：
{params_text}

约束说明章节内容：
{constraints_text}

严格按以下 JSON 数组格式返回，不要添加任何其他文字：
[
  {{
    "param_name": "x1",
    "platform": "Atlas A2 训练系列产品/Atlas A2 推理系列产品",
    "mode": "positive",
    "dtypes": ["FLOAT32", "FLOAT16", "BFLOAT16"]
  }},
  {{
    "param_name": "input",
    "platform": "Atlas 训练系列产品",
    "mode": "negative",
    "dtypes": ["BFLOAT16", "HIFLOAT8"]
  }}
]

无 dtype 约束信息时返回: []
"""


SHAPE_TO_DIMENSIONS_PROMPT = """\
你是一个维度解析专家。将输入的 shape 描述列表解析为二维数组格式。

规则：
1. "1-8" → [[1,8]]
2. "(N,C,H,W)" 含4维 → [[null,null],[null,null],[null,null],[null,null]]
   每个维度对应一组 [min,max]，若无明确范围用 null
3. "2D" → 二维但无范围 → [[null,null],[null,null]]
4. 空字符串或无法解析 → []
5. "与输入相同" → []
6. "标量" 或 "0-D" → []
7. "1-D" → [[null,null]]
8. "[2, 3, 4]" 表示固定3维 → [[2,2],[3,3],[4,4]]

严格按 JSON 数组返回，每个 shape 对应一个元素，顺序与输入一致。
不要添加任何其他文字。

Shape 描述列表：
{shapes}"""


ALLOWED_RANGE_VALUE_BUILD_PROMPT = """\
你是一个参数取值范围解析专家。从文档内容中提取参数 "{param_name}" 的数值型取值范围。

规则：
1. 只提取数值型取值范围（上下界），忽略其他约束
2. "范围0-100" → [[0, 100]]
3. "取值范围[1, 8]" → [[1, 8]]
4. "大于0" → [[1, null]]（null表示无上界）
5. "小于1024" → [[null, 1023]]（null表示无下界，取前一个整数）
6. 无法确定范围 → []
7. 多个范围用逗号分隔 → [[0, 100], [1024, 2048]]
8. "枚举值: 1, 2, 3" → [[1, 1], [2, 2], [3, 3]]（每个枚举值视为单点范围）

严格按 JSON 二维数组返回，不要添加任何其他文字。

参数名: {param_name}
参数类型: {param_type}

文档内容:
{context_text}"""


RELATION_OBJECT_BUILD_PROMPT = """\
你是一个参数约束表达式生成专家。根据参数关系的自然语言描述，判断约束类型并生成形式化 Python 表达式。

## expr_type 枚举（必须严格选择其一）
- shape_broadcast：形状需满足广播关系
- shape_choice：形状可以是多个候选之一
- shape_equality：形状必须完全相同
- shape_dependency：shape 元素值之间的依赖
- shape_value_dependency：特定轴值之间的依赖
- value_dependency：张量元素值或参数值之间的约束
- presence_dependency：共存规则（如 A is None == B is None）
- type_dependency：数据类型依赖
- type_equality：数据类型必须一致
- format_equality：数据格式必须一致

## expr 生成规则
1. 输出合法 Python 布尔表达式，返回值为 bool
2. 形状引用：param.shape[dim_index]，如 x.shape[0]
3. 值引用：param.range_value，如 groups.range_value
4. 类型引用：param.dtype
5. 格式引用：param.format
6. 不允许 null；无法写出表达式时用空字符串 ""
7. 蕴含逻辑用 (B) if (A) else True
8. 等价逻辑用 (A) == (B)
9. 不要使用 if/else 自然语言语句
10. expr 中只能引用 params 列表中的参数
11. 不要使用 implies 这个词
12. 不要使用伪代码
13. 不要在 expr 中使用平台值作为判断条件
14. 涉及生成器表达式时必须包裹在 all() 或 any() 中，不允许 lambda

## 函数签名上下文
{signatures_text}

## 输入
relation_type（粗粒度提示）：{relation_type}
params：{params}
description：{description}
source_citation：{source_citation}

严格按以下 JSON 返回，不要添加任何其他文字：
{{"expr_type": "...", "expr": "..."}}
"""
