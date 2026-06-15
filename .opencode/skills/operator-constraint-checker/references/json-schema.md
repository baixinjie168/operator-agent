# JSON约束文档Schema参考

本文档描述算子约束JSON文件的**实际结构**，基于 `res/check0612/` 目录下的约束文档。

## 顶层结构

```json
{
    "operator_name": "aclnnBidirectionLSTMV2",
    "function_explanation": "该算子实现长短时记忆（LSTM）网络计算...",
    "product_support": ["Atlas 推理系列产品"],
    "function_signature": "aclnnStatus aclnnBidirectionLSTMV2GetWorkspaceSize(...)",
    "deterministic_computing": { "<产品名>": { "value": "true", "src_text": "..." } },
    "inputs": { ... },
    "outputs": { ... },
    "constraints_in_parameters": { "<产品名>": [ ... ] },
    "return_info": [ ... ],
    "dtype_support_description": {}
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `operator_name` | string | 算子名称 |
| `function_explanation` | string | 功能说明 |
| `product_support` | string[] | 支持的产品列表 |
| `function_signature` | string | 函数签名（C++原型） |
| `deterministic_computing` | dict | 确定性计算信息，按产品分组 |
| `inputs` | dict | 输入参数字典，key为参数名 |
| `outputs` | dict | 输出参数字典，key为参数名 |
| `constraints_in_parameters` | dict | 参数间约束，按产品分组，值为约束数组 |
| `return_info` | list | 返回值错误码信息 |
| `dtype_support_description` | dict | 数据类型支持描述 |

## inputs / outputs - 参数字典

每个参数以名称为key，值为**按产品名分组**的字典：

```json
{
    "x": {
        "Atlas 推理系列产品": {
            "description": "LSTM单元的输入向量，公式中的x。",
            "type": {
                "value": "aclTensor",
                "src_text": ""
            },
            "format": {
                "value": ["ND"],
                "src_text": ""
            },
            "is_optional": {
                "value": false,
                "src_text": ""
            },
            "is_support_discontinuous": {
                "value": true,
                "src_text": "√"
            },
            "is_operator_param": {
                "value": false,
                "src_text": ""
            },
            "dimensions": {
                "value": [[3, 3]],
                "src_text": "3"
            },
            "array_length": "N/A",
            "dtype": {
                "value": ["FLOAT16"],
                "src_text": ""
            },
            "allowed_range_value": {
                "value": [],
                "src_text": ""
            }
        }
    }
}
```

### 参数字段详解

| 字段 | 类型 | 含义 | 与Markdown对应 |
|------|------|------|---------------|
| `description` | string | 参数描述 | "描述"列 |
| `type.value` | string | 参数的C语言类型名（去掉const/\*/&修饰符） | 函数签名中的参数类型，如 aclTensor, aclScalar, aclTensorList, int64_t, bool |
| `type.src_text` | string | 原文引用 | |
| `format.value` | string[] 或 "N/A" | Tensor的数据格式（内存布局）；非Tensor参数为"N/A" | "数据格式"列，如 ["ND"]；非Tensor为 "-" |
| `is_optional.value` | bool | 是否可选（可省略/可传空） | 参数名含Optional或描述中提及 |
| `is_support_discontinuous.value` | bool 或 "N/A" | 是否支持非连续（不连续内存）的Tensor；非Tensor参数为"N/A" | "非连续Tensor"列，√/×/- |
| `is_operator_param.value` | bool | 是否是算子函数的参数。true=衍生参数（文档中某参数的取值被其他参数依赖，但该参数本身并非算子函数签名中的参数） | false=在参数表格中出现的算子参数；true=衍生参数(如batch_size) |
| `dimensions.value` | 见下方三种格式 | 参数shape的维度约束 | "维度(shape)"列 |
| `dimensions.src_text` | string | 维度原文 | 如 "3", "0-8", "标量", "-" |
| `array_length` | string | 数组类型参数（aclIntArray/aclTensorList等）的长度约束描述；非数组类型为"N/A" | 根据参数类型推断 |
| `dtype.value` | string[] | 参数允许的数据类型列表 | "数据类型"列 |
| `allowed_range_value.value` | list | 参数允许的取值范围 | 使用说明中的取值约束，如 [true, false], [[1,1]] |

### dimensions.value 三种格式

1. `[]` — 标量参数或无法确定维度（对应Markdown中的"-"或"标量"）
2. `[min_rank, max_rank]` — 维数范围，如 `[3, 3]` = 固定3维，`[0, 8]` = 0到8维（对应"3"或"0-8"）
3. `[[min, max], ...]` — 逐维范围，如 `[[1,1], [3,3]]` = 固定2维，第1维固定为1，第2维固定为3

注意：格式2和格式3可能同时出现在不同参数中，需根据具体情况判断。

### 衍生参数（is_operator_param = true）

某些参数如 `batch_size`, `hidden_size`, `input_size`, `time_step`, `total_batch_size` 是衍生参数：
- 文档中如果算子参数的取值依赖某个参数，但是这个参数并非是算子函数的参数，则 `is_operator_param` 为 `true`
- 在Markdown参数表格中可能不直接出现
- 但在"使用说明"文本和 `constraints_in_parameters` 表达式中被引用
- 这些参数的 `dimensions.value` 通常为 `[]`，`dimensions.src_text` 为 "标量"
- 维度1（参数完整性）检查时，衍生参数允许仅存在于JSON中而不出现在Markdown参数表格

### array_length 规则

- 数组类型参数（参数type为 `aclIntArray`、`aclTensorList`、`aclFloatArray` 等）：`array_length` 应有具体的长度约束描述或值
- 非数组类型参数（如 `aclTensor`、`aclScalar`、`int64_t`、`bool` 等）：`array_length` 应为 `"N/A"`

## constraints_in_parameters - 参数间约束

按产品名分组的约束数组：

```json
{
    "constraints_in_parameters": {
        "Atlas 推理系列产品": [
            {
                "expr_type": "shape_dependency",
                "expr": "wIh.shape[0] == 4 * initH.shape[-1] and wIh.shape[1] == x.shape[-1]",
                "relation_params": ["wIh", "initH", "x"],
                "src_text": "参数 wIh 使用说明：\"shape支持二维（4 * hidden_size, input_size）。\""
            }
        ]
    }
}
```

### 约束字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `expr_type` | string | 约束类型枚举 |
| `expr` | string | Python风格约束表达式 |
| `relation_params` | string[] | 涉及的参数名列表 |
| `src_text` | string | Markdown原文描述（验证依据） |

### expr_type 完整枚举

| expr_type | 含义 | 表达式示例 |
|-----------|------|-----------|
| `type_equality` | 数据类型一致性 | `x2.dtype == x1.dtype and x2.format == x1.format and x2.shape == x1.shape` |
| `type_dependency` | 条件类型依赖 | `((alpha.dtype in ['FLOAT32','DOUBLE']) if (x1.dtype in ['FLOAT32','BFLOAT16']) else ...)` |
| `shape_dependency` | shape间依赖 | `wIh.shape[0] == 4 * initH.shape[-1] and wIh.shape[1] == x.shape[-1]` |
| `shape_value_dependency` | shape依赖参数值 | `initH.shape[0] == (2 * numLayers if bidirection else numLayers)` |
| `shape_choice` | shape多选条件 | `(bidirection==False and packed==False and yOut.shape==...) or ...` |
| `value_dependency` | 参数值约束 | `numLayers.range_value == 1` |
| `presence_dependency` | 参数存在性依赖 | `(packed == False) if (batchSizeOptional is None) else True` |
| `self_shape_nonempty` | 自身shape非空检查 | `all(d > 0 for d in x1.shape)` |
| `self_dtype_consistency` | 自身dtype一致性 | `all(x1[i].dtype == x1[0].dtype for i in range(len(x1)))` |

### 表达式中的特殊语法

- `.shape[N]` — 参数的第N维大小
- `.dtype` — 参数的数据类型
- `.format` — 参数的数据格式
- `.range_value` — 参数的取值（用于衍生参数引用）
- `len(x.shape)` — 参数的shape长度（维度个数/ndim）。**注意**：Markdown中的"shape size"指的就是shape的长度（维度个数），不是元素总数。例如 x 的 shape 是 (B,C,T)，那么 x 的 shape size 是 3，而不是 B*C*T
- `is None` — 判断可选参数是否未提供
- 条件表达式：`A if condition else B`

## return_info - 返回值信息

```json
[
    {
        "return_value": "ACLNN_ERR_PARAM_NULLPTR",
        "error_code": 161001,
        "description": ["如果传入参数是必选输入，输出或者必选属性，且是空指针。"]
    }
]
```

## 检查时的关键对应关系

### 维度1（参数完整性）：Markdown参数表格 → JSON inputs/outputs

| 检查项 | 方法 |
|--------|------|
| 参数名 | Markdown参数表格中每行的"参数名"（排除框架参数）应在 JSON inputs 或 outputs 中出现 |
| 输入/输出分类 | Markdown中标记"输入"的 → JSON inputs 中；标记"输出"的 → JSON outputs 中 |
| 衍生参数 | JSON中 is_operator_param=true 的参数允许不在Markdown参数表格中出现 |
| 框架参数 | workspaceSize、executor 等框架参数不纳入完整性检查 |

### 维度2（参数属性）：Markdown参数表格列 → JSON参数字段

| Markdown列 | JSON字段 | 检查要点 |
|------------|----------|----------|
| 函数签名中的类型 | type.value | 去掉const/\*/&修饰符后一致。如 `const aclTensorList *x1` → `aclTensorList` |
| 数据类型 | dtype.value | 列表完全一致（顺序无关） |
| 数据格式 | format.value | 如 ["ND"] 对应 "ND"；非Tensor为 "N/A" 对应 "-" |
| 维度(shape) | dimensions.value + src_text | [0,8]对应"0-8"，[3,3]对应"3"，[]对应"-"或"标量" |
| 非连续Tensor | is_support_discontinuous.value | true对应√，false对应×，"N/A"对应"-" |
| 使用说明(取值) | allowed_range_value.value | 取值约束描述对应 |
| 参数名Optional后缀 | is_optional.value | 可选性一致 |
| (根据type推断) | array_length | 数组类型有具体值，非数组为"N/A" |

### 维度3（约束完整性）：Markdown使用说明 → constraints_in_parameters

| 使用说明模式 | 预期expr_type | 示例表达式 |
|-------------|--------------|-----------|
| "数据类型与入参x一致" | type_equality | `x2.dtype == x1.dtype` |
| "数据格式与入参x一致" | type_equality | `x2.format == x1.format` |
| "shape与x保持一致" | shape_dependency | `out.shape == x.shape` |
| "shape size大于等于x的shape size" | shape_dependency | `len(out.shape) >= len(x.shape)`（shape size = 维度个数，即 `len(shape)`） |
| "shape为[B, H]或[B, 1, H]" | shape_choice | 多分支or表达式 |
| "shape为[H]，H与x中H维一致" | shape_dependency | `w.shape[0] == x.shape[-1]` |
| "shape为(num_layers, batch, hidden)" | shape_value_dependency | `shape[0] == numLayers` |
| "当前只支持1" | value_dependency | `numLayers.range_value == 1` |
| "仅在bidirection为True时有效" | presence_dependency | `(p is None) if (not bidirection)` |
| "不支持空Tensor" | self_shape_nonempty | `all(d > 0 for d in x.shape)` |
| "该参数中所有Tensor的数据类型保持一致" | self_dtype_consistency | `all(x[i].dtype == x[0].dtype for i in range(len(x)))` |
| "数据类型与入参x1的数据类型具有对应关系" | type_dependency | 条件分支表达式 |
| 产品特定说明中的类型对应关系 | type_dependency | 条件分支表达式 |
| 公式隐含的列表长度相等 | (待补充) | `len(x1) == len(x2)` |

### 维度4（约束正确性）：四项子检查

| 子检查 | 检查内容 | 判定标准 |
|--------|---------|----------|
| relation_params完整性 | expr中引用的所有变量名是否都在relation_params中 | 遗漏→警告；不存在的变量名→失败 |
| src_text准确性 | src_text是否与Markdown原文语义一致 | 不一致→警告 |
| Python语法有效性 | expr是否为合法的Python表达式 | 语法错误→失败 |
| 描述一致性 | expr是否正确反映src_text的语义 | 语义不匹配→失败；expr_type分类不当→警告