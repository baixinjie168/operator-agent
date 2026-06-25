# 模式：数值范围提取

## 适用场景

参数取值为连续或离散的数值范围。
参数类型通常为 int64_t / int32_t / int8_t / double / float。

## 识别特征

- 描述中出现数值范围描述: "0-100", "[1,8]", "大于0", "0或1"
- 常见关键词: "取值范围", "范围", "大于", "小于", "不超过"

## 拆解规则

保留原文描述，不转换格式。type 设为 "range"。

### 格式对照表

| 原文描述 | allowed_range_value | type |
|---------|---------------------|------|
| "0-100" | "0-100" | range |
| "[1, 8]" | "[1, 8]" | range |
| "0或1" | "0或1" | range |
| "大于0" | "大于0" | range |
| "小于1024" | "小于1024" | range |
| "取值范围为0或1" | "0或1" | range |

## 示例

### 示例 1：明确范围

输入: "innerPrecise取值范围为0或1。"
输出:
```json
{"param_name": "innerPrecise", "platform": "", "allowed_range_value": "0或1", "type": "range"}
```

### 示例 2：区间范围

输入: "headNum取值范围[1, 32]"
输出:
```json
{"param_name": "headNum", "platform": "", "allowed_range_value": "[1, 32]", "type": "range"}
```

### 示例 3：半开区间

输入: "padding必须大于等于1"
输出:
```json
{"param_name": "padding", "platform": "", "allowed_range_value": "大于等于1", "type": "range"}
```

### 示例 4：枚举型整数

输入: "actType取值为0到5的整数"
输出:
```json
{"param_name": "actType", "platform": "", "allowed_range_value": "0-5", "type": "range"}
```
