---
name: operator-constraint-checker
description: 算子文档与约束文档一致性检查工具。用户提供算子Markdown文档路径和JSON约束文档路径，自动比对参数完整性、参数属性正确性、参数间约束完整性与正确性，生成带导航栏的明亮风格HTML分析报告。当用户提到"检查算子约束"、"比对算子文档"、"验证约束表达式"、"算子参数检查"、"constraint check"、"生成约束报告"时使用此skill。即使用户只是说"帮我看看这两个文件是否一致"或"检查一下这个算子的约束对不对"，也应该触发此skill。
---

# 算子文档与约束文档一致性检查

用户提供两个**文件路径**：算子Markdown文档路径 + JSON约束文档路径。
读取这两个文件，逐项比对一致性，生成明亮风格的HTML分析报告。

## 触发格式

用户通过以下格式触发此skill：

```
使用operator-constraint-checker skill处理
算子文档：<Markdown文件路径>
算子约束文档：<JSON文件路径>
输出路径：<输出目录或HTML文件路径>
```

**输入说明**：

- **算子文档**：算子Markdown文档的文件路径，如 `operators/nn/aclnnAdaLayerNorm.md`
- **算子约束文档**：JSON约束文档的文件路径，如 `res/check0612/aclnnBidirectionLSTMV2.json`
- **输出路径**：输出目录路径（如 `res/check0612`），报告文件名为 `{operator_name}_constraint_report.html`；也可以是带 `.html` 后缀的完整文件路径

用户输入的是**文件路径**，不是文档内容。需要先用Read工具读取文件内容，再进行分析。

## 工作流程

### Step 1: 读取文件

用Read工具分别读取两个文件的完整内容。

### Step 2: 解析Markdown文档

从Markdown文档中提取以下信息：

- **算子名称**：标题 `# xxx`
- **产品支持情况**：表格中每行的产品名和支持状态（√/×）
- **功能说明**：接口功能描述、计算公式（`$$`之间的LaTeX）
- **函数签名**：从代码块中提取 `aclnnXxxGetWorkspaceSize` 的完整参数列表（参数名 + C类型）
- **参数表格**：在"参数说明"章节的HTML表格中，表头为：
  `参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor`
  - 需去除 `<ul>`, `<li>`, `<code>` 等HTML标签获取纯文本
  - 注意 `rowspan`/`colspan`
  - 重点关注"使用说明"列，它包含参数间的关系约束描述
  - **区分输入参数和输出参数**：根据"输入/输出"列的值
  - **排除非算子参数**：`workspaceSize`、`executor` 等框架参数不纳入检查
- **约束说明**：确定性计算等
- **调用示例**：代码中的shape构造可辅助验证约束正确性
- **产品特定说明**：参数表格之后的文字描述（如按产品区分的scalar类型对应关系）

### Step 3: 解析JSON约束文档

JSON约束文档的完整结构详见 `references/json-schema.md`。核心结构如下：

**顶层字段**：
- `operator_name`：算子名称
- `function_explanation`：功能说明
- `product_support`：支持的产品列表（字符串数组）
- `function_signature`：函数签名
- `deterministic_computing`：确定性计算信息（按产品分组）
- `inputs`：输入参数字典
- `outputs`：输出参数字典
- `constraints_in_parameters`：参数间约束（按产品分组）
- `return_info`：返回值信息

**参数结构**（inputs/outputs 中的每个参数）：
参数以名称为key，值是**按产品名分组**的字典。每个产品下包含的字段及其**正确含义**如下：

| 字段 | 含义 | 类型 | 示例 |
|------|------|------|------|
| `type` | 参数的C语言类型名（去掉const/\*/&修饰符） | `{ "value": "aclTensor", "src_text": "" }` | aclTensor, aclScalar, aclTensorList, int64_t, bool |
| `format` | Tensor的数据格式（内存布局） | `{ "value": ["ND"], "src_text": "" }` | ["ND"], "N/A"（非Tensor参数） |
| `is_optional` | 该参数是否可选（可省略/可传空） | `{ "value": true/false, "src_text": "" }` | true, false |
| `is_support_discontinuous` | 是否支持非连续（不连续内存）的Tensor | `{ "value": true/false, "src_text": "√" }` | true, false, "N/A"（非Tensor参数） |
| `dimensions` | 参数shape的维度约束 | `{ "value": ..., "src_text": "3" }` | 见下方三种格式 |
| `array_length` | 数组类型参数的长度约束；非数组类型为"N/A" | string | "N/A", 具体值 |
| `dtype` | 参数允许的数据类型列表 | `{ "value": ["FLOAT16"], "src_text": "" }` | ["FLOAT32", "FLOAT16"] |
| `allowed_range_value` | 参数允许的取值范围 | `{ "value": [], "src_text": "" }` | [true, false], [[1,1]] |
| `is_operator_param` | 是否是算子函数的参数。文档中某些参数的取值被其他参数依赖，但该参数本身并非算子函数签名中的参数 | `{ "value": true/false, "src_text": "" }` | true（衍生参数）, false（算子参数） |

**dimensions.value 三种格式**：
1. `[]` — 标量参数或无法确定维度
2. `[min_rank, max_rank]` — 维数范围，如 `[3, 3]` = 固定3维，`[0, 8]` = 0到8维
3. `[[min, max], ...]` — 逐维范围，如 `[[1,1], [3,3]]` = 固定2维，第1维固定为1，第2维固定为3

**参数间约束**（constraints_in_parameters）：
按产品名分组的数组，每条约束包含：
- `expr_type`：约束类型（见下方枚举）
- `expr`：Python风格表达式
- `relation_params`：涉及的参数名列表
- `src_text`：Markdown原文描述

**expr_type 枚举**：
| expr_type | 含义 | 示例 |
|-----------|------|------|
| `type_equality` | 数据类型一致性 | `x2.dtype == x1.dtype` |
| `type_dependency` | 条件类型依赖 | 当x1为BFLOAT16时alpha为FLOAT32 |
| `shape_dependency` | shape间依赖关系 | `wIh.shape[0] == 4 * initH.shape[-1]` |
| `shape_value_dependency` | shape依赖参数值 | `initH.shape[0] == 2 * numLayers if bidirection else numLayers` |
| `shape_choice` | shape多选条件 | 根据bidirection/packed组合选择不同shape |
| `value_dependency` | 参数值约束 | `numLayers.range_value == 1` |
| `presence_dependency` | 参数存在性依赖 | 逆向参数仅在bidirection=True时有效 |
| `self_shape_nonempty` | 自身shape非空 | `all(d > 0 for d in x1.shape)` |
| `self_dtype_consistency` | 自身dtype一致 | Tensor列表中所有元素dtype一致 |

### Step 4: 执行四维度检查

检查按以下四个维度依次执行，每个维度独立生成检查结果。

#### 维度1：参数完整性检查

**目标**：检查JSON的 `inputs`/`outputs` 中的参数是否与Markdown参数表格一致，是否有遗漏或多余。

**检查方法**：

1. 从Markdown参数表格中提取**所有算子参数**（排除 `workspaceSize`、`executor` 等框架参数）
2. 从JSON的 `inputs` + `outputs` 中提取**所有参数名**
3. 逐一对比：
   - Markdown中有但JSON中缺失的参数 → **失败（遗漏）**
   - JSON中有但Markdown中没有的参数 → 检查 `is_operator_param`：
     - 若 `is_operator_param = true`（衍生参数）→ **通过**（衍生参数允许不在Markdown参数表中）
     - 若 `is_operator_param = false` → **失败（多余参数）**
   - 参数名大小写不一致 → **失败**
4. 检查输入/输出分类是否正确：
   - Markdown中标记为"输入"的参数应在 `inputs` 中
   - Markdown中标记为"输出"的参数应在 `outputs` 中

**输出**：参数完整性表格，列出每个参数的存在状态（JSON有/Markdown有/遗漏/多余）。

#### 维度2：参数属性正确性检查

**目标**：对每个参数的各属性字段，逐一检查JSON值是否与Markdown文档一致。

**检查方法**：

对每个参数，按以下字段逐一对比JSON与Markdown：

| JSON字段 | Markdown来源 | 检查规则 |
|----------|-------------|----------|
| `type.value` | 函数签名中的参数类型 | 去掉const/\*/&修饰符后应与C类型一致。如签名中 `const aclTensorList *x1` → type应为 `aclTensorList` |
| `format.value` | "数据格式"列 | `["ND"]` 对应 "ND"，非Tensor参数为 "N/A" 对应 "-" |
| `is_optional.value` | 参数名含Optional后缀 / 使用说明中提及可选 | `true` 对应可选，`false` 对应必选 |
| `is_support_discontinuous.value` | "非连续Tensor"列 | `true` 对应 "√"，`false` 对应 "×"，"N/A" 对应 "-" |
| `dimensions.value` | "维度(shape)"列 | 三种格式对应关系见下方 |
| `dimensions.src_text` | "维度(shape)"列原文 | 应保留原文，如 "3", "0-8", "标量" |
| `array_length` | 参数类型推断 | 数组类型（aclIntArray, aclTensorList等）应有具体值或描述；非数组类型为 "N/A" |
| `dtype.value` | "数据类型"列 | 类型列表必须完全一致（顺序无关），如 ["FLOAT32", "FLOAT16", "INT32"] |
| `allowed_range_value.value` | "使用说明"中的取值约束 | 如 "[true, false]" 对应bool参数，"[[1,1]]" 对应"只支持1" |

**dimensions 对应规则**：
| Markdown "维度(shape)"列 | dimensions.value | 说明 |
|--------------------------|------------------|------|
| "-" 或 标量 | `[]` | 标量参数无维度 |
| "3" | `[3, 3]` 或 `[[3, 3]]` | 固定3维 |
| "0-8" | `[0, 8]` | 0到8维 |
| "2-4" | `[2, 4]` | 2到4维 |

**is_operator_param 检查**：
- `is_operator_param = true` 的参数（衍生参数如 batch_size, hidden_size）：在Markdown参数表格中可能不直接出现，但会在使用说明文本中被引用
- `is_operator_param = false` 的参数：必须在Markdown参数表格中出现
- 检查衍生参数是否被其他参数的约束表达式所引用，若未被引用则可能是多余的

**输出**：按产品分组的参数属性检查表格，每个参数每个字段一行。

#### 维度3：参数间约束完整性检查

**目标**：检查Markdown中描述的参数间关系是否在JSON `constraints_in_parameters` 中都有对应的约束表达式，是否存在遗漏。

**检查方法**：

1. **遍历Markdown中所有约束来源**：
   - 每个参数"使用说明"列中的关系性描述
   - 参数表格后的产品特定说明
   - 计算公式中隐含的约束
   - 调用示例中体现的隐含约束

2. **逐条匹配JSON约束**：
   - 对于每条Markdown约束描述，在JSON `constraints_in_parameters` 中查找对应的约束（通过 `src_text` 匹配或语义匹配）
   - 未找到对应约束的 → 标记为**缺失**

3. **常见缺失模式**：
   - "数据类型与入参x一致" → 应有 `type_equality` 约束
   - "shape与x保持一致" → 应有 `shape_dependency` 约束
   - "不支持空Tensor" → 应有 `self_shape_nonempty` 约束
   - "该参数中所有Tensor的数据类型保持一致" → 应有 `self_dtype_consistency` 约束
   - 公式中列表长度相等 → 应有列表长度一致性约束
   - 产品特定说明中的类型对应关系 → 应有 `type_dependency` 约束

**输出**：缺失约束表格，列出Markdown原文、涉及参数、建议补充的表达式。

#### 维度4：参数间约束正确性检查

**目标**：对JSON中每条约束，从以下四个方面检查其正确性。

**检查方法**：

对 `constraints_in_parameters` 中的每条约束，执行四项子检查：

##### 4.1 relation_params 完整性

检查 `relation_params` 列表是否包含了表达式 `expr` 中引用的**所有参数名**：
- 解析 `expr` 中出现的所有变量名（如 `x1`, `alpha`, `out` 等）
- 对比 `relation_params` 列表
- 遗漏的参数 → **警告**
- 使用了不存在的参数名（在 `inputs`/`outputs` 中找不到） → **失败**

##### 4.2 src_text 原文准确性

检查 `src_text` 是否准确引用了Markdown原文：
- `src_text` 应与Markdown中对应参数的"使用说明"或产品特定说明的文字语义一致
- `src_text` 中描述的约束对象（参数名）是否与 `relation_params` 一致
- `src_text` 中是否有修正说明（如"注：原文误写为..."）暗示原文本身有问题

##### 4.3 表达式Python语法有效性

检查 `expr` 是否为合法的Python表达式：
- 语法是否正确（括号匹配、运算符合法、条件表达式完整）
- 使用的属性访问是否正确：`.dtype`, `.shape`, `.format`, `.range_value`
- `is None` 判断是否用于可选参数
- 列表推导/生成器表达式语法是否正确
- 字符串字面量的引号是否正确

##### 4.4 表达式与描述一致性

检查 `expr` 是否正确反映 `src_text` 的语义：
- **type_equality**：dtype/format/shape 的比较是否完整（src_text提到几个就检查几个）
- **type_dependency**：条件分支是否覆盖 src_text 中描述的所有数据类型映射
- **shape_dependency**：shape索引是否正确（`shape[-1]` 最后一维 vs `shape[0]` 第一维），算术关系是否正确
- **shape_value_dependency**：算术关系是否与 src_text 一致（如 `4 * hidden_size`）
- **shape_choice**：条件分支是否完整覆盖 src_text 中描述的所有组合
- **value_dependency**：取值约束是否正确（如 `.range_value == 1` 对应"只支持1"）
- **presence_dependency**：可选参数的 `is None` 检查是否完整
- **self_shape_nonempty**：`all(d > 0 for d in x.shape)` 格式是否正确
- **self_dtype_consistency**：TensorList内dtype一致性表达是否正确
- **expr_type 分类**：`expr_type` 是否与约束的实际语义匹配（如混合了类型一致性和shape约束的不应该标为 `value_dependency`）
- **"shape size" 术语**：Markdown中的"shape size"指的是 shape 的长度（即维度个数/ndim），**不是**元素总数。例如 x 的 shape 是 (B,C,T)，那么 x 的 shape size 是 3，而不是 B*C*T。对应的表达式应使用 `len(x.shape)` 表示

**输出**：每条约束的四项子检查结果表格，列出表达式、四项子检查的结论和分析详情。

### Step 5: 生成HTML报告

生成单文件HTML，所有CSS内联，无需外部资源。
详细模板和样式规范见 `references/html-template.md`。

**报告风格**：明亮风格
- 白色/浅灰背景，清晰的边框和阴影
- 明亮的渐变色页头
- 鲜明的颜色编码标签（绿=通过，橙=警告，红=失败，蓝=信息）
- 清晰的表格和代码块
- 充足的内边距和行间距

**报告结构**：
1. 页头（算子名称和报告标题）
2. 固定导航栏（4个检查维度 + 结论，每项显示计数徽标）
3. 摘要统计栏（通过/警告/失败总数）
4. 算子信息卡片
5. 4个检查章节（每个含表格和分析详情）
6. 综合结论
7. 导航栏高亮脚本（IntersectionObserver）

### Step 6: 确定输出路径

从用户输入中提取输出路径：
- 如果用户提供的是目录路径（如 `res/check0612`），则输出文件为 `{目录}/{operator_name}_constraint_report.html`
- 如果用户提供的是带 `.html` 后缀的完整文件路径，则直接使用该路径
- 算子名称从JSON的 `operator_name` 字段提取
- 如果提取不到输出路径，默认：`./{operator_name}_constraint_report.html`

## 注意事项

1. **平台维度**：JSON的 `inputs`/`outputs` 和 `constraints_in_parameters` 都按产品名分组，检查时需针对每个支持的产品分别验证。如果多个产品的检查结果完全一致，可以在报告中合并展示并标注
2. **衍生参数**：JSON中 `is_operator_param: true` 的参数（如 batch_size, hidden_size, time_step）是衍生参数，在Markdown中可能不直接出现在参数表格中，而是出现在使用说明文本里。维度1检查时需特殊处理
3. **src_text对照**：每条约束的 `src_text` 字段保存了Markdown原文描述，是验证表达式正确性的直接依据
4. **两段式接口**：检查时主要关注 `GetWorkspaceSize` 接口的参数约束。`workspaceSize`、`executor` 是框架参数，不纳入参数检查
5. **表达式语言**：JSON中的 `expr` 使用Python风格表达式，按Python语义理解。`.range_value` 表示引用参数的取值
6. **可选参数**：参数名含Optional的参数，在表达式中需检查 `is None` 判断是否完整
7. **函数签名解析**：从 `function_signature` 或 Markdown代码块中提取参数列表时，需要去掉 `const`、`*`、`&` 修饰符来获取纯类型名
8. **array_length**：仅对数组类型参数（如 `aclIntArray`、`aclTensorList`、`aclFloatArray` 等）有意义，非数组类型应为 "N/A"
9. **"shape size" 含义**：Markdown中出现的"shape size"指的是 shape 的长度（维度个数/ndim），而非元素总数。例：x 的 shape 是 (B,C,T)，则 x 的 shape size = 3（不是 B×C×T）。在表达式中用 `len(x.shape)` 表示
