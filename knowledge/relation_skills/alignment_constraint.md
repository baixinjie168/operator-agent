# 模式：对齐/整除约束（类型 4）

## 适用场景
参数间需要满足整除、对齐或倍数关系。常见于通信域大小与输入维度的约束。

## 识别特征
- "必须能被X整除" / "需要是X的整数倍"
- "按X对齐" / "必须是X的倍数"
- "X必须能整除Y"

## 模板
- 整除: {param_A} % {param_B} == 0
- 对齐: {param_A} % {align_size} == 0

## 示例
输入: "BS必须能被rankSize整除", params=["BS", "rankSize"]
输出: {"expr_type": "alignment", "expr": "BS % rankSize == 0"}

输入: "K必须是quantGroupSize的整数倍", params=["K", "quantGroupSize"]
输出: {"expr_type": "alignment", "expr": "K % quantGroupSize == 0"}

输入: "n为8的整数倍", params=["n"]
输出: {"expr_type": "self_alignment", "expr": "n % 8 == 0"}
