# 模式：参数自身约束（类型 3）

## 适用场景
单个参数的取值范围、枚举值、维度限制等。
仅涉及一个参数，不涉及其他参数间的依赖。

## 识别特征
- relation_type 为 self_constraint
- params 仅包含 1 个参数
- description 描述该参数自身的限制

## 子模式

### 取值范围
识别："取值范围为 245~333" / "大于0" / "0到1之间" / "取值在...之间"
模板：
  {min} < {param}.range_value < {max}
  {param}.range_value > {min}
  {param}.range_value < {max}

示例：
  输入: "opSize的取值范围为245~333", params=["opSize"]
  输出: {"expr_type": "self_value_range",
         "expr": "245 < opSize.range_value < 333"}

  输入: "groups 必须大于0", params=["groups"]
  输出: {"expr_type": "self_value_range",
         "expr": "groups.range_value > 0"}

### 允许值枚举
识别："只支持 0, 1, 2" / "取值为 true/false" / "可选值为..."
模板：{param}.range_value in [{v1}, {v2}, ...]

示例：
  输入: "mode 只支持 0, 1, 2", params=["mode"]
  输出: {"expr_type": "self_value_enum",
         "expr": "mode.range_value in [0, 1, 2]"}

### 维度数量限制
识别："支持 1~8 维" / "维度不超过 4" / "维度数..."
模板：
  len({param}.shape) >= {min}
  len({param}.shape) <= {max}
  {min} <= len({param}.shape) <= {max}

示例：
  输入: "x 支持 1~8 维", params=["x"]
  输出: {"expr_type": "self_shape_dim_range",
         "expr": "1 <= len(x.shape) <= 8"}

### shape 各维度值限制
识别："每个维度不超过 2^31" / "各维大小..."
模板：all(d <= {max} for d in {param}.shape)

### 空 Tensor 限制
识别："不支持空 Tensor" / "不允许空 Tensor"
模板：all(d > 0 for d in {param}.shape)
