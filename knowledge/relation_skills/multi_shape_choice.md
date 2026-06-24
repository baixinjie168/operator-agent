# 模式：多 Shape 候选（类型 4）

## 适用场景
参数的 shape 可以是多个候选之一，选择条件取决于其他参数的值
或枚举参数的取值。

## 识别特征
- description 包含多个 shape 选项（如 [E, N1]/[N1]）
- 或 description 包含 "或" / "分别" / "当...时为"
- 不同条件下 shape 不同

## 表达式模板
# 二选一（条件驱动）
({target}.shape == [shape_A]) if (condition) else ({target}.shape == [shape_B])

# 多选一（枚举驱动）
({target}.shape == [shape_A]) if ({enum}.range_value == "mode_A")
else ({target}.shape == [shape_B]) if ({enum}.range_value == "mode_B")
else True

# unless 结构（条件不满足时约束生效）
not({condition}) or ({target}.shape == [{vars}])

## 示例
输入: "per-channel时为[E, N1]，per-tensor时为[N1]"
  params: ["bias2Optional"]
  implicit_vars: E, N1
输出:
  expr_type: "shape_value_dependency"
  expr: (bias2Optional.shape == [E.range_value, N1.range_value])
        if (quantization_type.range_value == "per-channel")
        else (bias2Optional.shape == [N1.range_value])
        if (quantization_type.range_value == "per-tensor")
        else True

输入: "当 axis 为 0 时 x 的 shape 为 (N, C)，为 1 时为 (C, N)"
  params: ["x", "axis"]
输出:
  expr_type: "shape_value_dependency"
  expr: (x.shape == [N.range_value, C.range_value])
        if (axis.range_value == 0)
        else (x.shape == [C.range_value, N.range_value])
        if (axis.range_value == 1)
        else True
