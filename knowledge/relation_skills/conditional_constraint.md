# 模式：条件约束（类型 4）

## 适用场景
一个约束在特定条件下才生效。通常由枚举参数或参数的存在性触发。

## 识别特征
- "当X为Y时..." / "如果X=Y，则..."
- "X不为空时..." / "X为空时..."
- "在...场景下，X需要满足..."

## 模板
- 枚举条件: (condition) implies (constraint)
  Python 表达: (cond_expr) if (cond) else True
  或: all(constraint for ... if condition)
- 存在性条件: (param is not None) implies (constraint)

## 示例

### 枚举条件约束
输入: "当activation为geglu/swiglu/reglu时，N1=2*K2", params=["activation", "N1", "K2"]
输出:
  expr_type: "conditional"
  expr: "activation.range_value in ['geglu','swiglu','reglu'] implies N1 == 2*K2"
  Python: "(N1 == 2*K2) if activation.range_value in ['geglu','swiglu','reglu'] else True"

输入: "当activation为gelu/fastgelu/relu/silu时，N1=K2", params=["activation", "N1", "K2"]
输出:
  expr: "(N1 == K2) if activation.range_value in ['gelu','fastgelu','relu','silu'] else True"

### 存在性条件约束
输入: "offset不为空时，scale shape为[E,1,N]", params=["offset", "scale"]
输出:
  expr: "(scale.shape == [E, 1, N]) if offset is not None else True"

输入: "offset为空时，k要求为quantGroupSize的整数倍且k<=18432", params=["offset", "k", "quantGroupSize"]
输出:
  expr: "(k % quantGroupSize == 0 and k <= 18432) if offset is None else True"

### 场景条件约束
输入: "splitItem为2/3的场景，out长度不等于1", params=["splitItem", "out"]
输出:
  expr: "(len(out) == 1) if splitItem in [2, 3] else True"

## 注意事项
- "implies" 不是 Python 关键字，用 `(consequent) if (antecedent) else True` 表达
- 条件中的枚举值用字符串列表: `activation.range_value in ['geglu', 'silu']`
- 存在性判断用 `is not None` / `is None`

## 可选参数 None 守卫（重要）

当约束引用了可选参数（名称含 Optional）的 .shape/.dtype/.format 属性时，
**必须**用 `(expr) if (paramName is not None) else True` 包装，
防止参数为 None 时属性访问异常。

### 示例

输入: "biasDequant2Optional 的 shape[0] 等于 N", params=["biasDequant2Optional"]
输出:
  expr_type: "shape_value_dependency"
  expr: "(biasDequant2Optional.shape[0] == N.range_value) if biasDequant2Optional is not None else True"

输入: "scaleOptional 的数据类型与 x 一致", params=["scaleOptional", "x"]
输出:
  expr_type: "type_equality"
  expr: "(scaleOptional.dtype == x.dtype) if scaleOptional is not None else True"

注意: 如果约束本身就是存在性判断（presence_dependency），无需额外包装。
