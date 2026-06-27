# 参数别名映射知识库

> 约束表达式中的简写参数名 → 实际参数名映射。
> 约束抽取（LLM 生成）时，对成对出现的参数可能使用简写名（如 `weight` 代替 `weight1`/`weight2`），
> 导致 `constraints_in_parameters` 中的 `expr` 和 `relation_params` 引用了参数列表中不存在的名字。
> 本文件记录这些简写名到实际参数名的映射，供约束审查 Agent 和程序化解析工具使用。

## 映射语义

| 映射类型 | YAML 格式 | 含义 |
|----------|-----------|------|
| **重命名**（1对1） | `expertTokens: [expertTokensOptional]` | 简写名等同于单个实际参数，直接替换 |
| **广播AND**（1对多） | `weight: [weight1, weight2]` | 简写名约束需展开为每个实际参数都满足（AND 组合） |

## 查询优先级

```
算子专属映射 > 全局默认映射 > 无映射（参数名即实际名）
```

## 全局默认映射

适用于所有算子。算子专属映射会覆盖同名的全局默认。

```yaml
# 暂无全局默认映射，后续发现跨算子通用的简写模式时在此补充
```

## 算子专属映射

### aclnnFFNV3

```yaml
expertTokens: [expertTokensOptional]
deqScaleOptional: [deqScale1Optional, deqScale2Optional]
weight: [weight1, weight2]
biasOptional: [bias1Optional, bias2Optional]
antiquantScaleOptional: [antiquantScale1Optional, antiquantScale2Optional]
antiquantOffsetOptional: [antiquantOffset1Optional, antiquantOffset2Optional]
```

**说明**：
- `expertTokens` 是 `expertTokensOptional` 的简写（漏了 Optional 后缀）。
- `deqScaleOptional` / `weight` / `biasOptional` / `antiquantScaleOptional` / `antiquantOffsetOptional` 是对成对参数（1/2 后缀）的泛化简写，约束需同时适用于两个实际参数。
