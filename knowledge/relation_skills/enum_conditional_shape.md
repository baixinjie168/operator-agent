# 模式：枚举条件 + 条件 Shape（类型 4）

## 适用场景
当关系描述中同时包含：
- 枚举参数取特定值（如 quantization_type == "per-channel"）
- 参数存在性判断（如 expertTokensOptional 不为空 / 有专家）
- 条件 shape（如 [E, N1] / [N1]）

## 识别特征
- description 包含 "per-channel" / "per-tensor" / "per-group"
- description 包含 "有/无专家" / "不为空" / "可选"
- shape 描述中出现方括号 [E, N1] 格式
- 涉及 quantization_type 隐式枚举参数

## 拆解规则
1. 枚举条件 -> quantization_type.range_value == "per-channel"
2. 存在性条件 -> expertTokensOptional is not None
3. 条件 shape -> bias2Optional.shape == [E.range_value, N1.range_value]
   其中 E、N1 是隐式维度变量，用 var.range_value 引用
4. 布尔结构：
   - "在...时为 X" -> (X) if (条件) else True
   - "有/无...时分别为 X/Y" -> not(条件1) or (X)
     即：条件不满足时约束不生效

## 表达式模板
# 单条件
not({enum_param}.range_value == "{value}")
  or ({target}.shape == [{vars}])

# 双条件（枚举 + 存在性）
not({enum_param}.range_value == "{value}"
    and {presence_param} is not None)
  or ({target}.shape == [{vars}.range_value, ...])

## 示例
输入: "per-channel下输入在有/无专家时分别为[E, N1]/[N1]"
  params: ["bias2Optional", "expertTokensOptional"]
  implicit_vars: E=bias2Optional.shape[0], N1=bias2Optional.shape[1]

输出:
  expr_type: "shape_value_dependency"
  expr: not(quantization_type.range_value == "per-channel"
           and expertTokensOptional is not None)
       or (bias2Optional.shape == [E.range_value, N1.range_value])

## 示例（FFNV3 有专家/无专家 + per-channel/per-tensor）

输入: "weight1 的 shape: 有专家时为 [E, K1, N1]，无专家时为 [K1, N1]；
       per-channel 时 antiquantScaleOptional 为 [E, 1, N1]，per-tensor 时为 [1]"
  params: ["weight1", "antiquantScaleOptional", "expertTokensOptional"]
  implicit_vars: E, K1, N1 (命名维度变量)

输出:
  expr_type: "shape_value_dependency"
  expr: "((weight1.shape == [E.range_value, K1.range_value, N1.range_value]) if expertTokensOptional is not None else (weight1.shape == [K1.range_value, N1.range_value])) and ((antiquantScaleOptional.shape == [E.range_value, 1, N1.range_value]) if quantization_type.range_value == 'per-channel' else (antiquantScaleOptional.shape == [1]) if quantization_type.range_value == 'per-tensor' else True) if antiquantScaleOptional is not None else True"

注意:
- 有专家时 weight1 为 3 维 [E, K1, N1]，无专家时为 2 维 [K1, N1]
- per-channel/per-tensor 条件由 quantization_type.range_value 驱动
- antiquantScaleOptional 是可选参数，需加 is not None 守卫
- E/K1/N1 是隐式维度变量，用 .range_value 引用
