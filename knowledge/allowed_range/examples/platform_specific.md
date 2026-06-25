# 模式：平台相关取值

## 适用场景

不同平台支持不同的取值集合，文档中按平台分行描述。

## 识别特征

- 出现 "Atlas A2 训练系列产品"、"Atlas 推理系列加速卡产品" 等平台前缀
- 不同平台用冒号或换行分隔
- 每个平台有自己的取值范围

## 拆解规则

每个平台产出一个独立的 JSON 对象，platform 字段标注平台名称。
平台名称保持原文中的完整名称。

## 示例

### 示例 1：activation 平台差异

输入:
```
Atlas A2 训练系列产品/Atlas A2 推理系列产品：当前支持fastgelu/gelu/relu/silu以及geglu/swiglu/reglu。
Atlas 推理系列加速卡产品：当前支持fastgelu/gelu/relu/silu。
```

输出:
```json
[
  {"param_name": "activation", "platform": "Atlas A2 训练系列产品/Atlas A2 推理系列产品", "allowed_range_value": "fastgelu,gelu,relu,silu,geglu,swiglu,reglu", "type": "enum"},
  {"param_name": "activation", "platform": "Atlas 推理系列加速卡产品", "allowed_range_value": "fastgelu,gelu,relu,silu", "type": "enum"}
]
```

### 示例 2：维度范围平台差异

输入:
```
Atlas A2 训练系列产品：支持输入的维度最少是2维，最多是8维。
Atlas 推理系列加速卡产品：支持输入的维度是2维。
```

输出:
```json
[
  {"param_name": "x", "platform": "Atlas A2 训练系列产品/Atlas A2 推理系列产品", "allowed_range_value": "2-8", "type": "range"},
  {"param_name": "x", "platform": "Atlas 推理系列加速卡产品", "allowed_range_value": "2", "type": "range"}
]
```
