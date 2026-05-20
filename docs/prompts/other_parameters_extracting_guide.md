# other_parameters 提取规则

本文件定义了从 CANN 算子说明文档 中提取 **other_parameters ** 的严格规则。

## 目标输出数据结构
```python 
class ParameterType(str, Enum):
    """参数类型枚举"""
    ACL_TENSOR = "aclTensor"
    ACL_SCALAR = "aclScalar"
    ACL_INT_ARRAY = "aclIntArray"
    ACL_FLOAT_ARRAY = "aclFloatArray"
    ACL_BOOL_ARRAY = "aclBoolArray"
    ACL_TENSOR_LIST = "aclTensorList"
    ACL_SCALAR_LIST = "aclScalarList"
    ACL_OP_EXECUTOR = "aclOpExecutor"
    ACL_RT_STREAM = "aclrtStream"
    # 标量类型
    DOUBLE = "double"
    BOOL = "bool"
    UINT64_T = "uint64_t"
    INT64_T = "int64_t"
    INT32_T = "int32_t"
    FLOAT = "float"
    FLOAT16 = "float16"
    VOID = "void"
    SIZE_T = "size_t"
    
class OtherParameterConstraint(BaseModel):
    platform: str = Field(..., description="平台名称")
    value: List[List[float]] | List[float] | List[List[int]] | List[int] | List[str] | List[bool] = Field(
        default_factory=list, description="参数的允许或禁止的值")
    rule: str = Field(default="", description="未在'函数原型'中定义的参数需要满足的约束条件")

    model_config = {"extra": "forbid"}

class OtherParameters(BaseModel):
    name: str = Field(..., description="参数名称")
    type: str = Field(..., description="参数类型")
    description: str = Field(default=None, description="参数描述")
    constraints: List[OtherParameterConstraint] = Field(..., description="参数相关约束")

    model_config = {"extra": "forbid"}

class OtherParametersOutput(BaseModel):
    """顶层模型"""
    other_parameters: List["OtherParameters"] = Field(default_factory=list, description="其他未在'函数原型'中定义的参数")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。


## other_parameters
`other_parameters` 属性是一个列表，其中包含未在主函数参数中定义的额外参数。这些参数通常是在算子文档中引入的，用于描述关系或约束，但并不属于实际函数签名的一部分。

### 提取规则：
- **识别**：
 - 只有文档明确把某个符号量当作约束符号、维度符号、计数符号或中间定义量时，才将其提取到 `other_parameters`。
 - 只有函数原型之外、且后续约束表达必须引用的中间符号才进入 `other_parameters`，例如文档明确定义的 `r` 为输入 rank。
 - 在 “功能说明”“约束说明”“参数说明中的使用说明”“公式说明”“shape 说明” 等区域查找未在“参数说明”或函数原型中定义、但被稳定引用的符号。
 - 例如，在 `shape 为 (max(tpWorldSize, 1) * A, H)` 这类描述中，`A` 和 `H` 可以作为候选额外参数；但如果某个量只是一次性公式中的临时中间表达式，则不要提取。
 - 不要把真实函数参数的别名或公式里的临时写法抽成额外参数；例如 `groups` 已是函数参数时，不要再提取 `g`。
 - 示例 shape 中的 `A/B/C/D` 这类占位维度通常只是说明性符号，不是可独立建模的额外参数；没有明确约束用途时应返回 `[]`。
- **结构**：
 - 列表中的每个元素应为一个对象，包含以下字段：
 - `name`：参数名称，为字符串类型。
 - `description`：参数的定义性描述，只解释“这个符号代表什么”。
 - `type`：参数类型，参考 `ParameterType`。
 - `constraints`：约束对象的列表，每个约束对象包含：
 - `platform`：执行平台（例如 `'Atlas A3 训练系列产品/Atlas A3 推理系列产品'`、`'Atlas A2 训练系列产品/Atlas A2 推理系列产品'`、`'Atlas 200I/500 A2 推理产品'`、`'Atlas 推理系列产品'`、`'Atlas 训练系列产品'` 或 `'All'`）。
 - `rule`：该符号与张量 shape、dtype、维度位置或其他符号之间的关系规则。
 - `value`：该符号的取值限制。
 - 被提取的中间符号必须有清晰的 `type`、`description` 和可复用的 `rule/value`，以便 `parameter_constraints` 或 `inter_parameter_constraints` 引用。
- **平台处理**：
 - 每个 `constraints` 元素只对应一个平台范围。
 - 不要在单个元素中内嵌 `if platform == xxx` 这类平台分支逻辑。
 - 如果多个平台约束相同，可将 `platform` 设为 `All`；如果平台差异不同，则必须拆成多个元素。
- **description 字段**：
 - `description` 只负责定义符号含义，不要把 shape/dtype/range 关系堆进 `description`。
 - 如果文档中的关系被写进了 `description`，而 `rule` 为空或过于简化，应把这些关系迁回 `rule`。
- **rule 字段**：
  - `rule` 用于表达该符号与其他参数、维度或符号之间的关系；凡是和张量 shape、dtype、维度位置、rank 场景有关的关系，都应放在 `rule` 中。
  - 对 B、S、H、N、C、Hin、Hout 等中间符号，要覆盖其与相关张量所有关键维度的对应关系，不要只写某一种 rank 的下标关系。
  - 如果文档隐含了缺失的中间符号但确实需要它来表达约束，应先补齐该符号，再用 `rule` 建立对应关系。
  - `rule` 不允许为 `null`；如果没有关系规则，使用空字符串 `""`。
  - 不要把长段自然语言说明、平台分支或一次性临时表达式写进 `rule`。
- **value 字段**：
 - `value` 描述参数的取值限制，遵循与 `allowed_values.value` 相同的规则。
 - 对于数值参数：直接列出允许值，例如“取值为0和1” -> `[0, 1]`。
 - 对于范围参数：使用区间表达，例如“0到epWorldSize-1之间的整数” -> `[[0, epWorldSize-1]]`。
 - 对于分段参数：使用多个区间元素，例如“0到7168之间的整数或8192到10240之间的整数” -> `[[0, 7168], [8192, 10240]]`。
 - 不要把关系规则、条件判断、函数调用或自然语言说明写进 `value`。
- **通用规则**：
 - 不得重复同一个参数名。
 - 不要把真实函数参数、平台名、公式片段或纯说明文字误抽为 `other_parameters`。
 - 如果没有任何符合条件的额外参数，则该字段应为 `[]`。
