# 模式：数据类型一致性（类型 4）

## 适用场景
多个参数要求数据类型相同。常见于输入/输出参数间、或量化参数与主输入参数间。

## 识别特征
- "与X的数据类型一致" / "和X数据类型相同"
- "数据类型与X保持一致"
- "dtype与X一致"

## 模板
{param_A}.dtype == {param_B}.dtype

## 示例
输入: "bias的数据类型与weight一致", params=["bias", "weight"]
输出: {"expr_type": "dtype_equality", "expr": "bias.dtype == weight.dtype"}

输入: "gradOutput的数据类型，与self保持一致", params=["gradOutput", "self"]
输出: {"expr_type": "dtype_equality", "expr": "gradOutput.dtype == self.dtype"}
