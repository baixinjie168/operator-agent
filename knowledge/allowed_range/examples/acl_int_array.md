# 模式：aclIntArray 特殊取值

## 适用场景

参数类型为 aclIntArray，取值为特定的数组值（如 [-2, -1]）或空数组。
type 设为 "enum"。

## 识别特征

- 参数类型为 aclIntArray
- 描述中出现 "支持配置空或者[-2,-1]" 等特定数组值描述
- 常见关键词: "支持配置", "空", "或者"

## 拆解规则

将各个可选数组值用逗号分隔。
type 设为 "enum"。

## 示例

### 示例 1：可选数组值

输入: "支持配置空或者[-2,-1]"
输出:
```json
{"param_name": "alltoAllAxesOptional", "platform": "", "allowed_range_value": "空,[-2,-1]", "type": "enum"}
```

### 示例 2：多个特定数组

输入: "支持配置[-2,-1]或[-1,-2]或空"
输出:
```json
{"param_name": "axesOptional", "platform": "", "allowed_range_value": "[-2,-1],[-1,-2],空", "type": "enum"}
```
