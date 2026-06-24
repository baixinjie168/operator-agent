"""Prompts for specialized extraction: return codes, determinism, dtype combinations."""

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

{platform_context}规则：
1. 表头中每个列名映射到参数名（去掉"数据类型"后缀和反引号）
2. 每一行数据是一个完整的合法组合
3. 单元格含 "/" 时（如 "UINT64/INT64"），保留为字符串值，不要拆分
4. 单元格含 "null" 时，该参数在组合中省略（不出现在对象中）
5. 仅输出 JSON，不要添加任何其他文字
6. 如果有多个平台各自有一张子表，为每个平台分别输出一个对象
7. 若整张表没有标注平台，platform 必须设为上方适用平台列表中的第一个平台名称，不要设为"common"

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

无平台限定且上方无适用平台列表时，platform 才设为 "common"。
"""


DTYPE_CONSTRAINT_TEXT_PROMPT = """\
你是一个 CANN 算子数据类型组合提取专家。从下面的「约束说明」章节中，提取每个场景下的【完整参数 dtype 组合】。

约束说明通常按"场景"（如非量化场景、量化场景、伪量化场景）描述，每个场景内多个参数的数据类型是绑定约束（必须同时成立）。你必须把每个场景输出为一行完整组合，绝不能把参数拆成各自独立的清单再交叉组合。

{platform_context}规则：
1. 每一行（row）是一个【完整且合法的组合】：该场景下所有出现数据类型的参数，其 dtype 在同一行内同时取定。
2. 同一场景内绑定的参数（如"量化场景：x 为 INT8、weight 为 INT8、biasOptional 为 INT32"）必须放进同一行；禁止把不同场景的取值混搭到一行。
3. 参数"多值"有两种情况，必须严格区分，否则后端会错误交叉组合：
   3a. 单个参数【独立】有多种可取 dtype（如"deqScale 支持 UINT64、INT64、FLOAT32"）：在该参数上用 "/" 连接写成单个值（如 "UINT64/INT64/FLOAT32"），后端会独立展开成多行。
   3b. 多个参数要求【数据类型保持一致/相同】（如"deqScale1Optional 与 deqScale2Optional 数据类型保持一致"）：禁止给这些参数各自写 "/"（否则后端会把它们各自独立展开，生成两者不一致的非法组合）；必须改为按共享值拆成多行，每行内这些参数取同一个值（如一行都为 UINT64，下一行都为 INT64，再一行都为 FLOAT32）。
4. 不同场景拆成不同行；条件不同（如有专家/无专家、per-tensor/per-channel、y 为 FLOAT16/BFLOAT16）也拆成不同行。
5. 只在「数据类型」层面提取，忽略 shape、数据格式、连续性等其它约束。
6. 文档中未提及数据类型的参数（或仅与 shape/format 相关的参数）从 row 中省略。
7. 识别平台限定：按平台分别输出对象；若约束文字未明确标注平台名称，platform 必须设为上方适用平台列表中的第一个平台名称，不要设为 "common"；仅当上方无适用平台列表时 platform 才设为 "common"。
8. 如果约束说明中没有任何数据类型相关信息，返回空数组 []。

参数列表（用于参考参数名，只提取列表中存在的参数）：
{params_text}

约束说明章节内容：
{constraints_text}

严格按以下 JSON 格式返回（与组合表相同结构），不要添加任何其他文字：
[
  {{
    "platform": "平台名称",
    "rows": [
      {{"x": "INT8", "weight1": "INT8", "weight2": "INT8", "bias1Optional": "INT32", "scaleOptional": "FLOAT32", "offsetOptional": "FLOAT32", "y": "FLOAT16", "deqScale1Optional": "UINT64", "deqScale2Optional": "UINT64"}},
      {{"x": "INT8", "weight1": "INT8", "weight2": "INT8", "bias1Optional": "INT32", "scaleOptional": "FLOAT32", "offsetOptional": "FLOAT32", "y": "FLOAT16", "deqScale1Optional": "INT64", "deqScale2Optional": "INT64"}},
      {{"x": "FLOAT16", "weight1": "INT8", "weight2": "INT8", "antiquantScale1Optional": "FLOAT16", "antiquantOffset1Optional": "FLOAT16", "y": "FLOAT16"}}
    ]
  }}
]

上方示例中 deqScale1Optional 与 deqScale2Optional 要求"保持一致"，故按 3b 拆成多行（每行两者同值），禁止写成 "UINT64/INT64" 各自带 "/"。
无 dtype 组合信息时返回: []
"""
