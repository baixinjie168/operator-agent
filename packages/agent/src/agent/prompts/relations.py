"""Prompts for parameter relation/constraint expression building."""

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

注意：`parameter_representation` 类型由确定性节点生成（用于声明
命名变量与算子参数 shape 维度的对应关系，如
`BS.range_value == x1.shape[0]`），**LLM 不应产出此类型**。若要
表达多参数之间的派生约束（例如 output.shape[0]*rankSize == x1.shape[0]），
请使用 shape_value_dependency。

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
17. 当约束涉及命名维度变量（如 b、m、k 等出现在 shape 描述中的字母命名维度）时，
    使用对应 tensor 的 shape 索引表达，不要引入命名变量作为独立参数引用：
    - 例如 b 在 self 中是第 0 维 → self.shape[0]
    - n1 在 mat2 中是第 1 维 → mat2.shape[1]
    - 对于已知常量（如 k0=16, n0=16），直接使用数值
    - 如果上下文中提供了 Shape 维度映射表，参照其中的对应关系
18. "shape size"（或"shape 大小"、"shape 长度"、"维度数"）指的是 shape 的
    **维数（rank）**，即 len(x.shape)，而不是各维大小的乘积（元素总数）。
    例如 x 的 shape 是 (B, C, T)，则 x 的 shape size 是 3，不是 B*C*T。
    表达式中用 len(x.shape) 表示 shape size
19. 禁止使用 tuple()；需要类型转换时用 list() 代替，或直接比较
    （如 x.shape == y.shape，无需任何转换）
20. 平台外部常量（如 rankSize）可直接在 expr 中以 rankSize.range_value 形式引用，
    不需要将其列入 params 列表。外部常量的取值范围由平台决定，
    在 expr 中不要硬编码具体数值。约束表达式中如需引用外部常量，
    只需将实际涉及的算子参数列入 params，外部常量名不列入

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

### 示例 7: shape_value_dependency（命名维度变量 → tensor.shape[i]）
输入: description="self 的 k 与 mat2 的 k1 满足 ceil(k, k0) = k1，其中 k0 = 16",
      params=["self", "mat2"]
输出: {{"expr_type": "shape_value_dependency", "expr": "math.ceil(self.shape[2] / 16) == mat2.shape[2]"}}
注意：
- k 对应 self.shape[2]（self 的 shape 是 (b, m, k)，k 是第 2 维）
- k1 对应 mat2.shape[2]（mat2 的 shape 是 (b, n1, k1, k0, n0)，k1 是第 2 维）
- k0 = 16 是常量，直接写 16
- params 中不包含 b, m, k, n1 等命名维度变量

### 示例 8: shape_broadcast（特定维度的 broadcast）
输入: description="self 的 b 与 mat2 的 b 满足 broadcast 关系",
      params=["self", "mat2"]
输出: {{"expr_type": "shape_broadcast", "expr": "self.shape[0] == mat2.shape[0] or self.shape[0] == 1 or mat2.shape[0] == 1"}}
注意：b 在 self 中是 shape[0]，在 mat2 中也是 shape[0]

### 示例 9: shape_value_dependency（camelCase 参数名 + 标量参数）
输入: description="initH 的 shape 为 (numLayers, batch_size, hidden_size)，bidirection 为 True 时第一维为 2 * numLayers",
      params=["initH", "numLayers", "bidirection", "batch_size", "hidden_size"]
输出: {{"expr_type": "shape_value_dependency", "expr": "(initH.shape[0] == (2 * numLayers.range_value if bidirection.range_value else numLayers.range_value)) and initH.shape[1] == batch_size.range_value and initH.shape[2] == hidden_size.range_value"}}
注意：
- numLayers、batch_size、hidden_size 是函数签名中的真实标量参数（不是 shape 维度变量），
  引用其值时使用 .range_value
- numLayers 是 params 中的原始名称（camelCase），不要转换为 num_layers

### 示例 10: shape_dependency（shape size / 维度数比较）
输入: description="out 的 shape size 大于等于 x1 的 shape size", params=["out", "x1"]
输出: {{"expr_type": "shape_dependency", "expr": "len(out.shape) >= len(x1.shape)"}}
注意：
- "shape size" 指维数（rank），用 len(x.shape) 表示
- 不是各维大小的乘积，不要写成 out.shape[0]*out.shape[1]*... 的形式

## 函数签名上下文
{signatures_text}

## 参数 shape 信息
{param_shapes_text}

{implicit_params_text}
## 输入
relation_type（粗粒度提示）：{relation_type}
params：{params}
description：{description}
source_citation：{source_citation}

严格按以下 JSON 返回，不要添加任何其他文字：
{{"expr_type": "...", "expr": "...", "confidence": "high/medium/low", "uncertainty_reason": "不确定原因（仅当 confidence != high 时）"}}
"""
