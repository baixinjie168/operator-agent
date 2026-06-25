# 提示词优化模式

本文档汇总 operator-agent 项目中常见的 LLM 提示词问题模式与对应的修复模板。

## 优化原则

1. **示例驱动**：LLM 对 few-shot 示例响应最好，每个 prompt 应有 ≥ 3 个示例
2. **格式约束**：明确输出 schema（JSON 结构 + 字段类型），避免 LLM 输出自然语言
3. **边界覆盖**：列出常见的边界情况，让 LLM 知道怎么处理
4. **互斥规则**：明确禁止的输出（如 null、空字符串、嵌套结构不规范）
5. **可执行性**：约束表达必须是 Python 表达式（可被 eval() 执行）

## 常见模式速查

| 问题模式 | 影响的 prompt | 关键修复 |
|----------|---------------|----------|
| dimensions.value 出现 None | SHAPE_EXTRACT_PROMPT | 显式列出 3 种合法格式 |
| dtype.value 为空列表 | DTYPE_EXTRACT_PROMPT | 增加 dtype 默认值规则 |
| 隐式参数漏提 | IMPLICIT_PARAM_EXTRACT_PROMPT | 增加常见隐式参数清单 |
| constraints_in_parameters 为空 | PARAM_RELATION_EXTRACT_PROMPT | 要求每行使用说明至少对应一条 relation |
| expr 不是合法 Python | BUILD_PARAM_RELATIONS_PROMPT | 要求 expr 必须可被 eval() |
| format 字段为空 | DFORMAT_EXTRACT_PROMPT | 明确 Tensor=ND, 标量=N/A |
| array_length 缺失 | ARRAY_LENGTH_EXTRACT_PROMPT | 明确数组类型必须有 array_length |
| optional 误标 | OPTIONAL_EXTRACT_PROMPT | Optional 后缀 + "可省略" 关键词 |
| allowed_range 缺失 | ALLOWED_RANGE_EXTRACT_PROMPT | "只支持 X" / "取值 Y 到 Z" 必须提取 |
| 函数签名解析错误 | FUNCTION_SIGNATURE_EXTRACT_PROMPT | 强调去掉 const/*/& 修饰符 |

## 优化模板

### 模板 A：增强输出格式约束

适用场景：LLM 输出格式不固定，导致下游解析失败。

**修改前**：

```markdown
请提取参数 x 的维度约束。
```

**修改后**：

```markdown
请提取参数 x 的维度约束。

### 输出格式约定（必须遵守）
- 标量参数：dimensions.value = []
- 固定 N 维：dimensions.value = [N, N]  (例如 3 维 → [3, 3])
- 范围维度（如 0-8 维）：dimensions.value = [0, 8]
- 逐维范围（如"第一维固定为 1，第二维范围 [3, 3]"）：dimensions.value = [[1,1], [3,3]]

### 禁止的输出
- 字符串："3 维"
- null
- 浮点数：[3.0, 3.0]
- 嵌套不规范的列表：[3, [3, 3]]

### 示例
输入：参数 x 的维度是 0-8 维
输出：{"dimensions": {"value": [0, 8], "src_text": "0-8 维"}}

输入：参数 y 是标量
输出：{"dimensions": {"value": [], "src_text": "标量"}}
```

### 模板 B：增加覆盖率要求

适用场景：LLM 提取的参数/约束数量少于预期。

**修改前**：

```markdown
提取参数表中的所有参数。
```

**修改后**：

```markdown
提取参数表中的所有参数（含隐式参数）。

### 隐式参数识别清单（即便不在函数签名中也必须提取）
- alpha、beta、gamma（缩放/移位参数）
- axis、dim、axes（轴向参数）
- keepdim、transpose、broadcast（布尔开关）
- padding、stride、dilation、kernel_size（卷积相关）
- numLayers、bidirectional、batch_first（RNN 相关）

### 覆盖率要求
- 参数表每一行都必须有对应条目
- 使用说明列中提到的所有参数都必须有条目
- 输出 JSON 后必须自检：参数数量 == 参数表行数 + 使用说明中提到的额外参数数
```

### 模板 C：互斥规则

适用场景：LLM 输出矛盾的字段（如输入 INT8，输出 FLOAT32）。

**修改前**：

```markdown
提取 x 和 y 的 dtype。
```

**修改后**：

```markdown
提取 x 和 y 的 dtype。

### dtype 互斥规则
- 如果 y 与 x 同类型，y.dtype.value 必须等于 x.dtype.value
- 如果 y 与 x 类型不同，必须明确说明转换规则（如 x INT8 → y INT32）
- 当源文档说"取决于 X"时，必须列出 X 的所有可能 dtype

### 示例
输入：x 支持 FLOAT16/FLOAT32，y 是 x 的平方
输出：x.dtype = ["FLOAT16", "FLOAT32"]，y.dtype = ["FLOAT16", "FLOAT32"]（与 x 同类型）

输入：x 是 INT8，y 是累加结果
输出：x.dtype = ["INT8"]，y.dtype = ["INT32"]（累加器类型）
```

### 模板 D：可执行性要求

适用场景：LLM 输出的是自然语言而不是 Python 表达式。

**修改前**：

```markdown
提取参数间关系，生成 expr。
```

**修改后**：

```markdown
提取参数间关系，生成 expr。

### expr 输出规则
1. 必须是合法 Python 表达式，可用 eval() 执行
2. 属性访问：
   - 数据类型：x.dtype
   - 维度：x.shape / x.shape[0]
   - 数据格式：x.format
   - 取值：x.range_value
3. 比较：==、!=、<、>、in
4. 逻辑：and、or、not
5. 条件：if/else 三元表达式

### 合法 expr 示例
- `x.dtype == y.dtype` (类型一致)
- `x.shape[0] == 4 * hidden_size` (shape 依赖)
- `x.shape == y.shape` (shape 一致)
- `alpha.range_value == 1` (取值约束)
- `out is None if mode == 'train' else out.dtype == x.dtype` (条件类型)
- `all(d > 0 for d in x.shape)` (自身非空)

### 禁止输出
- 自然语言："x 与 y 类型相同"
- 伪代码：`x.shape = y.shape`
- 字符串：`"x.dtype == y.dtype"`
- 缺少运算符：`x.dtype y.dtype`
```

### 模板 E：边界条件覆盖

适用场景：参数是特殊情况（数组、可选、Tensor 列表）时容易出错。

**修改前**：

```markdown
提取 array_length。
```

**修改后**：

```markdown
提取 array_length。

### 必须有 array_length 的类型
- aclIntArray：array_length 必须是整数（如 2）或描述（如 "2-4"）
- aclFloatArray：同上
- aclBoolArray：同上
- aclTensorList：array_length 是 Tensor 数量
- aclScalarList：array_length 是 Scalar 数量

### 默认值
- 源文档未提及时，aclTensorList 默认 "1-8"（1 到 8 个 Tensor）
- aclIntArray 默认 "N/A"（由调用方决定）

### 示例
输入：x1 是 aclTensorList，包含 2 个 Tensor
输出：{"array_length": {"value": "2", "src_text": "包含 2 个 Tensor"}}

输入：pads 是 aclIntArray
输出：{"array_length": {"value": "N/A", "src_text": ""}}
```

## 自检 checklist

修改 prompt 后，用以下清单检查：

- [ ] 至少 3 个示例（覆盖常见 + 边界 + 互斥）
- [ ] 明确禁止的输出（null、空字符串、嵌套不规范）
- [ ] 明确字段类型（int / str / list / dict）
- [ ] 覆盖率要求（参数数量、relation 数量）
- [ ] 可执行性要求（expr 必须可 eval()）
- [ ] 引用源文档的关键词（如"只支持"、"与 X 同类型"）

## 验证方法

修改 prompt 后，按以下流程验证：

1. 选 3-5 个历史上失败的算子作为测试集
2. 重新跑约束提取 Pipeline
3. 用 `operator-iteration-analyzer` skill 再次分析
4. 对比修复前后的失败率

如果失败率下降 50% 以上 → 修改有效；如果不变 → 可能问题不在 prompt 而在代码逻辑。