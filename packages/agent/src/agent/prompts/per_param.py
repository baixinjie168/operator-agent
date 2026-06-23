"""Prompts for per-parameter attribute extraction: dtype, dformat, shape, optional, array_length, allowed_range."""

DTYPE_EXTRACT_PROMPT = """\
你是一个参数数据类型提取专家。从下面参数的描述文本中，提取其数据类型。

规则：
1. 查找描述中关于数据类型的信息，如 FLOAT16、FLOAT32、INT8、INT64 等
2. 常见数据类型示例：INT8, INT16, INT32, INT64, FLOAT16, FLOAT32, FLOAT64,
   BOOL, UINT8, STRING, TENSOR, SCALAR, LIST, BFLOAT16, COMPLEX64, COMPLEX128
3. 提取的数据类型值必须统一为大写，如"INT8"、"FLOAT32"、"TENSOR"
4. 如果有多个数据类型，用逗号分隔
5. 如果完全没有数据类型相关信息，dtype设为空字符串

严格按以下JSON格式返回，不要添加任何其他文字：
{{"param_name": "{param_name}", "dtype": "提取的大写数据类型或空字符串"}}

参数描述：
{params_text}
"""


OPTIONAL_EXTRACT_PROMPT = """\
你是一个参数可选性判断专家。从下面参数的描述文本中，判断该参数是否可选。

关键词清单（描述中出现以下任一表达即视为可选）:
- 中文: 可选、可空、可为空、可不传、不传、非必选、缺省值、默认值、默认
- 英文: Optional、optional、default、None、null、nullable
- 其他: 可以为 nullptr、可为 nullptr、空指针、未指定时

重要规则:
- 参数名为 "{param_name}"
- 参数名包含 Optional: {name_has_optional}
- 如果参数名包含 Optional，请忽略参数名，只看描述内容
- 如果描述内容中确实出现了上述关键词，返回 true，否则返回 false
- 请注意：如果描述中出现"必选"、"必须"、"不可为空"等字眼，则不是可选，返回 false

严格按以下JSON格式返回，不要添加任何其他文字:
{{"param_name": "{param_name}", "is_optional": true}}

参数描述:
{params_text}
"""


DFORMAT_EXTRACT_PROMPT = """\
你是一个参数数据格式提取专家。从下面参数的描述文本中，提取其数据格式（Data Format）。

规则：
1. 查找描述中关于数据格式的信息，如 ND、NCHW、NHWC 等
2. 常见数据格式示例：ND, NC, NCL, NCHW, NCDHW, NHWC, NWC, NC1HWC0, NC1HWC0_C4,
   FRACTAL_Z, FRACTAL_NZ, FRACTAL_ZZ, FRACTAL_ZN_LSTM, NDHWC, NDC1HWC0
3. 提取的数据格式值必须统一为大写
4. 如果有多个数据格式，用逗号分隔
5. 如果完全没有数据格式相关信息，dformat设为空字符串

严格按以下JSON格式返回，不要添加任何其他文字：
{{"param_name": "{param_name}", "dformat": "提取的数据格式或空字符串"}}

参数描述：
{params_text}
"""


SHAPE_EXTRACT_PROMPT = """\
你是一个参数shape提取专家。从下面参数的描述文本中，提取其维度(shape)信息。

规则：
1. 查找描述中关于维度/shape的信息，如"1D~8D"、"(N,C,H,W)"、"2D"等
2. 只有无条件、直接给出的shape值才提取
3. 如果shape依赖其他参数取值（出现"当..."、"如果..."等条件），则忽略
4. 提取的shape值应简洁，保留原文关键信息
5. 如果完全没有shape相关信息，shape设为空字符串

严格按以下JSON格式返回，不要添加任何其他文字：
{{"param_name": "{param_name}", "shape": "提取的shape值或空字符串"}}

参数描述：
{params_text}
"""


ARRAY_LENGTH_EXTRACT_PROMPT = """\
你是一个数组长度提取专家。从下面参数的描述文本中，提取该数组参数的长度/大小约束信息。

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
8. type 字段区分两种语义：
   - "range"（默认）：连续数值范围，如 "0-100"、"[-1, 1]"
   - "enum"：参数取值是特定的完整数组值（通常是 aclIntArray 类型），
     如 "支持配置空或者[-2,-1]" 中 [-2,-1] 是一个完整数组而非范围
   如果参数类型是 aclIntArray 且约束描述的是特定数组值，type 设为 "enum"

严格按以下 JSON 数组格式返回，不要添加任何其他文字:
[
  {{"platform": "平台名称或空字符串", "allowed_range_value": "取值范围描述", "type": "range或enum"}}
]

type 为 "range" 时可省略（默认 range）。无约束时返回: []

{semantic_rules_context}

文档章节内容:
{sections_text}
"""
