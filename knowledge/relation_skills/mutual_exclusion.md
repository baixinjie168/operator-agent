# 模式：多场景互斥存在性约束（类型 4）

## 适用场景
多个场景互斥（如非量化/量化/伪量化），每个场景下
一组参数的存在性有特定要求。约束须编码所有场景。

## 识别特征
- description 含 "非量化/量化/伪量化" 等互斥场景
- 涉及多组可选参数的存在性判断
- 场景间互斥（同一时刻只有一个场景生效）

## 因式分解规则（重要）
**禁止**生成超长 if-else 嵌套堆叠。
**必须**使用因式分解的简洁形式：

1. 提取公共因子：各场景共有的约束提取到外层
2. 场景列表用 any()/all() 编码
3. 每个场景的约束用 (cond, expr) 表示

## 表达式模板
# 三场景互斥（非量化/量化/伪量化）
# 每个场景下特定参数必须存在
# ⚠️ 条件必须基于 params 列表中的真实参数构造，禁止臆造场景变量名
not any([
    (非量化条件 and not (paramA is not None)),
    (量化条件   and not (paramB is not None)),
    (伪量化条件 and not (paramC is not None)),
])

# 或用蕴含形式（更清晰）
all(
    (paramX is not None) if cond else True
    for cond, paramX in [
        (非量化条件, paramA),
        (量化条件,   paramB),
        (伪量化条件, paramC),
    ]
)

## 示例（FFNV3 三场景互斥）
输入: "innerPrecise 为 0(非量化)时 deqScale1/2Optional 为 None；
       innerPrecise 为 1(量化)时 antiquantScale1/2Optional 为 None；
       innerPrecise 为 2(伪量化)时 deqScale1/2Optional 存在"
  params: ["deqScale1Optional", "deqScale2Optional",
           "antiquantScale1Optional", "antiquantScale2Optional",
           "innerPrecise"]

输出:
  expr_type: "presence_dependency"
  expr: "all(
    (deqScale1Optional is None and deqScale2Optional is None)
      if innerPrecise.range_value == 0 else True,
    (antiquantScale1Optional is None and antiquantScale2Optional is None)
      if innerPrecise.range_value == 1 else True,
    (deqScale1Optional is not None and deqScale2Optional is not None)
      if innerPrecise.range_value == 2 else True,
  )"

注意:
- 三场景互斥，用 all() + 条件蕴含编码
- 每个分支独立、无重复子表达式
- 总长度控制在 300 字符以内
- 场景判断条件必须用 params 中的真实参数构造（如 innerPrecise.range_value == N），
  禁止臆造 quantization / pseudo_quantization 等不在 params 列表中的变量名
