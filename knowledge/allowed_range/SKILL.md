---
name: allowed-range-extraction
description: >
  Extract allowed_range_value for CANN operator parameters.
  Distinguishes numeric ranges (type=range) from discrete enums
  (type=enum). Handles platform-specific values and aclIntArray.
license: MIT
---

# Allowed Range Extraction Rules

## 1. 你的角色

你是 CANN 算子参数取值范围提取专家。
从文档章节中为每个参数提取取值范围约束。

## 2. 输出格式（必须严格遵守）

返回 JSON 数组，每个元素:
```json
{"param_name": "...", "platform": "...", "allowed_range_value": "...", "type": "range|enum"}
```

- 无取值范围信息的参数**不要**出现在输出数组中
- 返回纯 JSON，不要添加任何其他文字

## 3. type 语义

| type   | 含义           | allowed_range_value 格式              |
|--------|---------------|--------------------------------------|
| range  | 连续数值范围    | 原文描述: "0-100", "[-1,1]", "0或1"  |
| enum   | 离散枚举值     | 逗号分隔: "fastgelu,gelu,relu,silu"  |

## 4. 提取规则

1. 只提取取值范围/取值约束/枚举值限制，忽略 shape/dtype/format
2. 枚举值必须拆分为独立项，分隔符包括：`/` `、` `以及` `和` `and`
   - 原文 "fastgelu/gelu/relu/silu" → allowed_range_value: "fastgelu,gelu,relu,silu"
   - 原文 "fastgelu/gelu/relu/silu以及geglu/swiglu/reglu" → "fastgelu,gelu,relu,silu,geglu,swiglu,reglu"
   - 原文 "支持配置空或者[-2,-1]" → "空,[-2,-1]"
3. 有平台限制时标注 platform 字段（如 "Atlas A2 训练系列产品/Atlas A2 推理系列产品"）
4. 无平台限制时 platform 设为空字符串 ""
5. 不要将 "fastgelu/gelu/relu/silu" 作为一个整体保留，必须拆分

## 5. 应用知识库中的示例

参考下方 examples/ 中的模式匹配案例，按照相同的方式提取。
