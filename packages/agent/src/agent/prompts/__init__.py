LLM_DESCRIPTION_EXTRACT_PROMPT = """\
你是一个参数信息提取专家。从下面的文档内容中，为参数 "{param_name}" 提取结构化属性信息，
并同时提取对应的原始文本作为溯源依据。

要求：
1. 仔细阅读文档内容，提取与参数 "{param_name}" 直接相关的所有信息
2. 只包含**平台无关的通用属性**，即所有平台都适用的约束
3. **不要包含**平台特定的约束，如：
   - "在 Atlas A2 平台下..."
   - "Atlas A3 支持..."
   - "仅 Atlas 200I 产品..."
   - 任何带有平台名称前缀的约束
4. **不要包含**与其他参数之间的关系约束，如：
   - "与参数X的数据类型一致"
   - "shape的第N维与参数Y相同"
   - "取值依赖参数Z"
   这类跨参数关系由其他模块专门处理，不要在描述中重复
5. 如果某个属性在文档中没有通用信息（只有平台特定信息或无信息），该节填"无"
6. 同时提取文档中与描述对应的原始句子/段落，作为 src_content 溯源依据

输出格式（llm_description 必须严格按以下 9 个章节组织，每个章节以 # 标题开头）：

# 角色
输入/输出

# 功能描述
参数的功能说明

# 数据类型
通用数据类型（如 FLOAT16、FLOAT32、INT8 等）

# 数据格式
通用数据格式（如 ND、NCHW、NHWC 等）

# 维度约束
通用维度/shape约束

# 是否支持非连续Tensor
是/否

# 使用条件
通用使用条件和约束（无平台前缀）

# 是否可选
必选/可选

# 取值范围
通用取值范围（无平台前缀）

严格按以下 JSON 格式返回，不要添加任何其他文字：
{{
  "llm_description": "按上述9个章节格式组织的结构化文本",
  "src_content": "对应的原始文档文本（保留原文措辞）",
  "direction": "输入/输出/未知",
  "is_support_discontinuous": true/false/null
}}

注意：
- direction: 判断参数是输入还是输出，如果无法判断则填"未知"
- is_support_discontinuous: 仅对 Tensor 类型参数有效，判断是否支持非连续 Tensor；非 Tensor 参数填 null
- llm_description 中每个章节标题必须使用 # 开头，章节之间用空行分隔
- 没有信息的章节内容填"无"，不要省略章节标题

文档内容：
{section_content}"""


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
你是一个维度解析专家。将输入的 shape 描述列表解析为维度约束数组。

规则：
1. "X-Y" 表示维度数量范围（支持 X 到 Y 维），返回 [X, Y]
   例如 "0-8" 表示支持 0 到 8 维 → [0, 8]
   例如 "1-8" 表示支持 1 到 8 维 → [1, 8]
2. 含符号维度（如 N, D_out 等非纯数字）→ 只返回维度数量，格式为 [数量, 数量]
   例如 "(N, D_out, H_out, W_out, 3)" 有 5 个维度 → [5, 5]
   例如 "(N, C, H, W)" 有 4 个维度 → [4, 4]
   例如 "5" 表示 5 维 → [5, 5]
3. "2D" 表示 2 维 → [2, 2]
4. "1-D" 表示 1 维 → [1, 1]
5. 空字符串或无法解析 → []
6. "与输入相同" 或 "same as input" → []
7. "标量" 或 "0-D" → []
8. 纯数字维度（无符号）→ 逐维度范围，格式为 [[min1,max1], [min2,max2], ...]
   例如 "[2, 3, 4]" 表示固定 3 维 → [[2,2], [3,3], [4,4]]

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
15. 当约束引用了以字母命名的维度（如"C代表类别数"、"T代表序列长度"），
    且该维度在参数的 shape 描述中始终处于固定语义位置（如"最后一维"），
    必须使用负索引 shape[-1] 而非固定正索引 shape[1] 或 shape[2]，
    因为同一参数可能有多种 shape 形式（如"(T,N,C)或(T,C)"），
    正索引在不同 shape 下指向不同的维度
16. 引用参数时必须使用 params 列表中的原始名称，不要自行转换命名风格
    （如不要将 camelCase 的 numLayers 转成 snake_case 的 num_layers）
17. 对于标量维度变量（如 time_step、batch_size、hidden_size 等非 Tensor 的
    隐含维度参数），引用其数值时必须使用 .range_value，
    例如 time_step.range_value，而非直接写 time_step
18. "shape size"（或"shape 大小"、"shape 长度"、"维度数"）指的是 shape 的
    **维数（rank）**，即 len(x.shape)，而不是各维大小的乘积（元素总数）。
    例如 x 的 shape 是 (B, C, T)，则 x 的 shape size 是 3，不是 B*C*T。
    表达式中用 len(x.shape) 表示 shape size

## 示例

### 示例 1: shape_equality
输入: description="x 和 y 的 shape 必须完全相同", params=["x", "y"]
输出: {{"expr_type": "shape_equality", "expr": "x.shape == y.shape"}}

### 示例 2: shape_broadcast
输入: description="x 和 y 的 shape 需满足广播关系", params=["x", "y"]
输出: {{"expr_type": "shape_broadcast", "expr": "all(x.shape[i] == y.shape[i] or x.shape[i] == 1 or y.shape[i] == 1 for i in range(len(x.shape)))"}}

### 示例 3: shape_value_dependency (条件逻辑 — 关键)
输入: description="当 scale 为 1 维时，其长度等于 x.shape[axis]", params=["x", "scale", "axis"]
输出: {{"expr_type": "shape_value_dependency", "expr": "(scale.shape[0] == x.shape[axis.range_value]) if len(scale.shape) == 1 else True"}}
注意：条件不满足时返回 True（约束仅在条件成立时生效）

### 示例 4: value_dependency (蕴含逻辑)
输入: description="如果 x 是 FLOAT16，则 y 也必须是 FLOAT16", params=["x", "y"]
输出: {{"expr_type": "value_dependency", "expr": "(y.dtype == 'FLOAT16') if (x.dtype == 'FLOAT16') else True"}}

### 示例 5: 全称量词
输入: description="x 的所有维度必须大于 0", params=["x"]
输出: {{"expr_type": "shape_dependency", "expr": "all(d > 0 for d in x.shape)"}}

### 示例 6: 无法形式化
输入: description="x 的取值需满足特定条件（详见说明）", params=["x"]
输出: {{"expr_type": "value_dependency", "expr": ""}}
注意：无法写出明确表达式时返回空字符串

### 示例 7: shape_value_dependency（隐含维度变量 + camelCase 参数名）
输入: description="initH 的 shape 为 (numLayers, batch_size, hidden_size)，bidirection 为 True 时第一维为 2 * numLayers",
      params=["initH", "numLayers", "bidirection", "batch_size", "hidden_size"]
输出: {{"expr_type": "shape_value_dependency", "expr": "(initH.shape[0] == (2 * numLayers.range_value if bidirection.range_value else numLayers.range_value)) and initH.shape[1] == batch_size.range_value and initH.shape[2] == hidden_size.range_value"}}
注意：
- numLayers 是 params 中的原始名称（camelCase），不要转换为 num_layers
- batch_size、hidden_size 是标量维度变量，引用其值时使用 .range_value
- numLayers 也是标量参数，引用其值时同样使用 .range_value

### 示例 8: shape_dependency（shape size / 维度数比较）
输入: description="out 的 shape size 大于等于 x1 的 shape size", params=["out", "x1"]
输出: {{"expr_type": "shape_dependency", "expr": "len(out.shape) >= len(x1.shape)"}}
注意：
- "shape size" 指维数（rank），用 len(x.shape) 表示
- 不是各维大小的乘积，不要写成 out.shape[0]*out.shape[1]*... 的形式

## 函数签名上下文
{signatures_text}

## 参数 shape 信息
{param_shapes_text}

## 输入
relation_type（粗粒度提示）：{relation_type}
params：{params}
description：{description}
source_citation：{source_citation}

严格按以下 JSON 返回，不要添加任何其他文字：
{{"expr_type": "...", "expr": "...", "confidence": "high/medium/low", "uncertainty_reason": "不确定原因（仅当 confidence != high 时）"}}
"""


LLM_DESCRIPTION_VERIFY_PROMPT = """\
你是一个参数描述审核员。请检查参数 "{param_name}" 的结构化描述是否存在信息遗漏。

参数名: {param_name}

当前描述:
{original_description}

疑似缺失的属性: {missing_attrs}

原始文档内容:
{document_context}

任务：
1. 仔细阅读原始文档，检查"疑似缺失的属性"是否在文档中存在相关信息
2. 如果存在：基于文档内容，将这些信息补充到对应章节中，生成增强版描述
3. 如果不存在：说明文档中确实没有这些信息

要求：
- 增强描述必须基于文档中的实际内容，不要编造
- 保持原有描述中正确的部分，只补充缺失的章节内容
- 仍然使用 # 标题的结构化格式，每个属性一个章节
- 仍然只包含平台无关的通用属性，不要加入平台特定约束
- 仍然不要包含跨参数关系约束（如"与X的dtype一致"）
- enhanced_description 必须比原始描述更完整才有意义，如果无法补充则 has_missing_info 设为 false

严格按以下 JSON 格式返回，不要添加任何其他文字：
{{
  "has_missing_info": true/false,
  "found_attrs": ["在文档中找到的缺失属性列表"],
  "enhanced_description": "增强后的完整结构化描述（仅当 has_missing_info 为 true 时提供）",
  "enhanced_src_content": "新增内容对应的原始文档文本"
}}"""
