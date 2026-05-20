# basic_info 提取规则

本文件定义了从 CANN 算子说明文档中提取 **basic_info（基本信息）** 的严格规则。

## 目标输出数据结构
```python 
class ApiFlow(str, Enum):
    """API流程类型枚举"""
    ONE_STEP = "one-step"
    TWO_STEP = "two-step"

class BasicInfoOutput(BaseModel):
    """顶层模型"""
    operation_name: str = Field(..., description="算子名称")
    description: str = Field(..., description="算子描述")
    api_flow: ApiFlow = Field(..., description="API流程类型: one-step 或 two-step")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

## 目标字段及提取顺序（必须按此顺序输出）

输出时必须严格按照以下字段顺序，且字段名称完全一致（包括中英文、括号）：

- operation_name(算子名称)
- description(算子描述)
- api_flow(阶段步骤)

## 提取规则（逐字段）

### 1. operation_name(算子名称)

- 从文档标题或第一行提取算子名称。

### 2. description(算子描述)

- 从"功能说明"章节提取算子功能描述，优先抽取明确对应“算子功能”或“接口功能”的句子。
- `description` 只保留功能说明本身，不要混入计算公式、参数解释、调用流程或其他模块内容。
- 如果“功能说明”和公式写在同一段，只截取功能语义，不要整段照搬；公式如 `out_i = ...`、数据范围、参数含义应交给其他模块表达。
- 变量名、下划线转义、大小写及占位符必须与原文保持一致。
- 如果文档没有稳定的功能说明句，宁可保守提取，也不要补写文档未直接给出的信息。
- 不要为了让描述更完整而补写文档未明确给出的推断内容，尤其不要把 `min/max`、`start/end/step`、示例 shape、参数解释或调用流程塞进基本描述。

### 3. api_flow(阶段步骤)

从"函数原型"章节判断 API 流程类型：

- **two-step**：存在两段式接口（先调用 GetWorkspaceSize 接口，再调用实际执行接口）
- **one-step**：单段接口，直接调用即可
- `api_flow` 只根据函数原型和调用流程判断，不要从描述性文字臆测阶段数。
