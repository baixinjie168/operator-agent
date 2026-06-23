"""Prompts for LLM-based parameter description extraction and verification."""

LLM_DESCRIPTION_EXTRACT_PROMPT = """\
你是一个参数信息提取专家。从下面的文档内容中，为参数 "{param_name}" 提取结构化属性信息，
并同时提取对应的原始文本作为溯源依据。

要求：
1. 仔细阅读文档内容，提取与参数 "{param_name}" 直接相关的所有信息
2. **数据类型、数据格式、维度约束**三个章节为"硬属性"章节，必须完整记录：
   - 优先提取平台无关的通用值（如"FLOAT16、BFLOAT16"、"ND"、"[M, K1]"）
   - 如果文档中只有平台特定的值（带有平台名称前缀），则**保留平台前缀并完整记录**
     每个平台的值，例如："Atlas A2 训练系列产品/Atlas A2 推理系列产品：[E, N2]/[N2]"
   - 必须完整收录该平台下的**所有子项**（如 per-channel、per-tensor、per-group 等），
     **不得遗漏任何一条**
   - 这三个章节只有在文档中**完全没有**对应信息时才填"无"
3. **使用条件、取值范围**等描述性章节只包含**平台无关的通用约束**：
   - 不要包含平台特定的约束，如"在 Atlas A2 平台下..."
   - 任何带有平台名称前缀的使用限制/取值约束都不要记录
4. **跨参数引用解析规则**（非常重要）：
   - 当文档中出现 "与xxx一致"、"和xxx相同"、"与`xxx`一致" 等引用时：
     - 如果上下文中**提供了被引用参数 xxx 的具体值**（如同一表格中 xxx 行
       的数据类型列写了 "FLOAT16、FLOAT、BFLOAT16"），则**解析引用**，
       将 xxx 的实际值填入对应属性。
       例如：posWeightOptional 的数据类型写 "与self一致"，而 self 的数据
       类型是 "FLOAT16、FLOAT、BFLOAT16"，则数据类型应填
       "FLOAT16、FLOAT、BFLOAT16"
     - 如果上下文中**没有提供被引用参数的具体值**，则该属性填"无"
   - 对于 **shape 维度的跨参数依赖**（如"shape与xxx一致"、"shape的第N维
     与参数Y相同"），这类约束由其他模块专门处理，在"维度约束"章节填"无"
   - 对于 **取值依赖**（如"取值依赖参数Z"），在"取值范围"章节填"无"
5. 如果某个属性在文档中完全没有信息，该节填"无"
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
- 对于**数据类型、数据格式、维度约束**三个"硬属性"章节：如果文档中存在平台特定的值
  （带有平台前缀），必须保留平台前缀完整记录，不得遗漏任何平台或子项（如 per-channel/per-group）
- 对于**使用条件、取值范围**等描述性章节，仍然只包含平台无关的通用属性，不要加入平台特定约束
- 仍然不要包含跨参数关系约束（如"与X的dtype一致"）
- enhanced_description 必须比原始描述更完整才有意义，如果无法补充则 has_missing_info 设为 false

严格按以下 JSON 格式返回，不要添加任何其他文字：
{{
  "has_missing_info": true/false,
  "found_attrs": ["在文档中找到的缺失属性列表"],
  "enhanced_description": "增强后的完整结构化描述（仅当 has_missing_info 为 true 时提供）",
  "enhanced_src_content": "新增内容对应的原始文档文本"
}}"""
