# 模式：张量内元素条件排序（类型 4 变体）

## 适用场景
某参数在特定条件下，其张量内元素需满足单调关系（非递减 / 非递增 / 严格升序 / 严格降序）。
通常由 bool 标志 + 存在性触发。

## 识别特征
- 后件："第 j 个元素 … 大于（或者）等于 … 第 i 个元素" / "非递减" / "升序" / "单调递增"
- 前件："X 为 true 且 … 时" / "X 不为空时" / "有专家时"

## 模板
```
(all(param[i] OP param[i + 1] for i in range(len(param) - 1))) if (cond) else True
```
OP 取值：
- 非递减：`<=`
- 非递增：`>=`
- 严格升序：`<`
- 严格降序：`>`

## 示例

### 输入
"tokensIndexFlag 为 true 且有专家（expertTokens 不为空）时，expertTokens 中的数值必须满足：
如果 i 和 j 都是 expertTokens 中有效的数组索引，且 j 大于 i，那么 expertTokens 中第 j 个
元素的数值大于或者等于 expertTokens 中第 i 个元素的数值。"

params=["expertTokensOptional", "tokensIndexFlag", "E"]

### 输出
```
expr_type: "self_value_ordering"
expr: "(all(expertTokensOptional[i] <= expertTokensOptional[i + 1]
         for i in range(len(expertTokensOptional) - 1)))
        if (expertTokensOptional is not None
            and len(expertTokensOptional) > 0
            and tokensIndexFlag.range_value == True
            and E > 0) else True"
```

## 注意
- **参数名映射**：源文用 "expertTokens"，实际参数为 `expertTokensOptional`。按实际签名映射
  （含 "expertTokens" 前缀的最长参数名）。
- **relation_params 必须完整**：expr 引用的所有参数（expertTokens*、tokensIndexFlag、E）
  都要写进 relation_params，否则 `validate_expr_refs` 会把 tokensIndexFlag/E 判为非法引用，
  且下游按 relation_params 做依赖传播会漏掉这两个依赖。
- **Optional None 守卫**：可选参数（名称含 Optional）必须用 `param is not None` 守卫；
  "不为空" 同时要求 `len(param) > 0`（仅 `is not None` 不够）。
- **implies 不是 Python 关键字**：用 `(consequent) if (antecedent) else True` 表达。
- 条件中的枚举值用字符串列表：`activation.range_value in ['geglu', 'silu']`（加 `.range_value`
  且字符串加引号，避免裸 Name 导致 NameError / `validate_expr_refs` 失败）。

## 确定性直出 vs LLM
源文措辞高度规整时（如 aclnnFFNV3 的 "第 j 个元素 … 大于或者等于 … 第 i 个元素"），
由 `constraint_extract.Pass 6c` 确定性正则直出，不经 LLM，最可靠。本技能供 LLM 路径
（complex_relation_agent）在遇到措辞变体时兜底。
