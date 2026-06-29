# 模式：TensorList 长度约束（类型 4）

## 适用场景
aclTensorList 参数的长度（元素个数）需要与其他参数一致或有特定限制。

## 识别特征
- "长度与X相同" / "长度和X一致"
- "个数与X相同" / "数量与X保持一致"
- "tensorList长度支持[1, N]"

## 模板
- 一致性: len({param_A}) == len({param_B})
- 范围: 1 <= len({param_A}) <= N

## 示例
输入: "biasOptional长度与weight相同", params=["biasOptional", "weight"]
输出: {"expr_type": "length_equality", "expr": "len(biasOptional) == len(weight)"}

输入: "tensorList长度支持[1, 128]", params=["x"]
输出: {"expr_type": "length_range", "expr": "1 <= len(x) <= 128"}

输入: "scaleOptional长度与weight相同。综合约束请参见约束说明。", params=["scaleOptional", "weight"]
输出: {"expr_type": "length_equality", "expr": "len(scaleOptional) == len(weight)"}
