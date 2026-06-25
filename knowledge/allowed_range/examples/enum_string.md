# 模式：字符串枚举拆分

## 适用场景

参数取值为离散字符串枚举（激活函数名、模式名、量化类型等）。
参数类型通常为 char* / const char*。

## 识别特征

- 参数类型为 char* / const char*
- 描述中出现 "/" 或 "、" 分隔的枚举值
- 常见关键词: "支持"、"当前支持"

## 拆解规则

分隔符包括: `/` `、` `以及` `和` `and`
必须拆分为独立项，用逗号分隔。

## 示例

### 示例 1：简单枚举

输入: "当前支持fastgelu/gelu/relu/silu"
输出:
```json
{"param_name": "activation", "platform": "", "allowed_range_value": "fastgelu,gelu,relu,silu", "type": "enum"}
```

### 示例 2：复合枚举（多种分隔符）

输入: "当前支持fastgelu/gelu/relu/silu以及geglu/swiglu/reglu"
输出:
```json
{"param_name": "activation", "platform": "", "allowed_range_value": "fastgelu,gelu,relu,silu,geglu,swiglu,reglu", "type": "enum"}
```

### 示例 3：量化类型枚举

输入: "量化类型，支持 per-channel/per-group/per-tensor/per-token"
输出:
```json
{"param_name": "quantization_type", "platform": "", "allowed_range_value": "per-channel,per-group,per-tensor,per-token", "type": "enum"}
```
