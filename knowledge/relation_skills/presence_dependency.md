# 模式：存在性依赖（类型 4）

## 适用场景
一个参数是否存在（为 None）取决于另一个参数的状态。

## 识别特征
- description 包含 "存在" / "不为空" / "可选" / "共存"
- relation_type 为 presence
- 涉及可选参数 (Optional)

## 表达式模板
# 互斥共存：A 和 B 要么都存在要么都不存在
(A is None) == (B is None)

# 条件存在：A 存在时 B 必须存在
(B is not None) if (A is not None) else True

# 条件不存在：A 存在时 B 必须不存在
(B is None) if (A is not None) else True

## 示例
输入: "expertTokensOptional 不为空时 bias2Optional 必须存在"
  params: ["expertTokensOptional", "bias2Optional"]
输出: {"expr_type": "presence_dependency",
       "expr": "(bias2Optional is not None) if (expertTokensOptional is not None) else True"}

输入: "x 和 y 必须共存，要么都存在要么都不存在"
  params: ["x", "y"]
输出: {"expr_type": "presence_dependency",
       "expr": "(x is None) == (y is None)"}
