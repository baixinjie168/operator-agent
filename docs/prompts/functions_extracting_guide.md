# functions 提取规则（函数/接口信息）

本文件定义了从 CANN 算子说明文档 中提取 **functions（函数信息）** 的严格规则。

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
    
class Parameter(BaseModel):
    """函数参数模型"""
    name: str = Field(..., description="参数名称")
    role: ParameterRole = Field(..., description="参数角色")
    type: str = Field(..., description="参数类型")
    is_optional: bool = Field(..., description="是否可选")
    description: str = Field(..., description="参数描述")
    format: Optional[Union[List[str], str]] = Field(default=None, description="Tensor格式")

    model_config = {"extra": "forbid"}

class OperatorFunction(BaseModel):
    """算子函数模型"""
    api_name: str = Field(..., description="API名称")
    description: str = Field(..., description="函数描述")
    parameters: List[Parameter] = Field(..., description="参数列表")

    model_config = {"extra": "forbid"}
    
class FunctionsOutput(BaseModel):
    """顶层模型"""
    functions: List["OperatorFunction"] = Field(..., description="函数列表")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

## 提取规则（逐字段）

- 如果文档中只有一个主要函数，则 `functions` 数组长度为 1。
- 如果存在多阶段接口（如 `GetWorkspaceSize` + 执行函数），则按出现顺序依次放入数组。

1. **识别函数原型区域**
   - 优先级顺序查找以下章节/关键词（任一命中即可）：
     - 函数原型 / `## 接口说明` / `## API 说明` / `## 函数签名`
     - 执行函数 / `## aclnnXXX` / `## aclopXXX`
     - 包含 `aclnn`、`aclop`、`ti_` 等前缀的代码块或加粗标题
     - 表格标题为“函数原型”或“参数说明”的表格上方内容
   - **常见模式**（至少包含以下之一）：
     - `函数名 + (参数列表)`
     - `返回类型 + 函数名 + (参数列表)`
     - 代码块中以 `aclnn` 开头的声明

2. **api_name（函数名称）**
   - 提取规则（优先级从高到低）：
     1. 小节标题（如 `### aclnnAdd`）
     2. 函数原型代码块的第一行中出现的函数名
     3. 参数表格上方的函数声明行
   - **要求**：
     - 只提取最主要的执行函数和必须的前置函数（如 `GetWorkspaceSize`）
     - 不要包含重载版本、内部实现函数、废弃函数
     - 保留完整名称（包括 `aclnn`/`aclop` 前缀）
   - **示例可接受值**：
     - `aclnnMatMul`
     - `aclnnAddLayerNormGetWorkspaceSize`
     - `aclnnFlashAttention`

3. **description（函数描述）**
   - 提取规则：
     - `description` 必须严格对应当前函数小节，不要把算子整体功能说明、公式说明或其他函数的信息搬进来。
     - 优先使用函数原型下方紧邻的一段话或官方一句话描述；如果没有独立描述，再从当前函数参数表格上方的说明文字中提炼。
     - 可以控制在 1-3 句，但必须保持原文语义，不要补入当前函数原型中没有出现的参数、流程或结果描述。
     - 二段式执行函数的 `description` 要描述当前函数本身；`GetWorkspaceSize` 返回的是 `workspaceSize` 和 `executor`，`workspace` 是调用方据此申请的，不要写成第一段接口直接返回 `workspace`。
     - 如果文档完全没有提供当前函数的独立描述，填写：`"文档中未提供函数描述"`。
   - **示例**：
     - "计算两个张量的逐元素加法，支持广播机制。"
     - "第一步：查询 Add + LayerNorm 操作所需的 workspace 大小并返回 executor。"

4. **parameters 数组（参数列表）**
   - 从md文档中的"函数原型"和"{OperatorName}GetWorkspaceSize"下的表格中提取该部分信息，或者从"{OperatorName}GetWorkspaceSize"下的"参数说明"提取该部分信息。
   - **4.1 name（参数名）**：直接从“参数名”列 / 参数列表中的变量名提取，保持原文大小写（如 `x`, `weight`, `epsilon`, `executor`）。参数名、数量和顺序必须严格对齐当前函数原型及对应参数说明表；不要把 `keepdim` 改成 `keepDim`，也不要凭语义新增、删除或拆出重复参数。
   - **4.2 role（输入/输出角色）**：根据文档中的明确标注判断（优先级从高到低）：
     - 列标题包含“输入/输出”或 “I/O” 时：
       - "输入" / "Input" / "I" → `"input"`
       - "输出" / "Output" / "O" → `"output"`
     - 若没有明确标注，必须结合函数原型、参数说明和调用语义综合判断，禁止只凭参数名猜测角色。
     - `role` 只能取 `input` 或 `output`。
     - 如果某个参数在文档中明确标注为“既是输入又是输出”（如原地更新、既读又写），仍统一标记为 `input`，并在 `description` 中保留“同时承担输出/回写语义”的说明；如果原文没有该语义，不要凭空添加。
     - 对“计算输入/计算输出”或原地更新参数，如果当前 schema 只能使用单值 `role`，应沿用项目允许的合法枚举值，并在 `description` 中保留原地更新或双向语义；`selfRef`、`inputRef` 等只保留一个形参，不要拆成 input/output 两个参数对象。
     - `workspace`、`workspaceSize`、`executor`、`stream` 等流程参数必须根据文档中的调用责任边界判断，不要把“调用方申请的输入”误标成输出，也不要把“接口返回的结果”误标成输入。

   - **4.3 type（参数类型）**：
     - 优先从“参数说明”中每个参数名称后的括号中提取，或者从“函数原型”中的参数定义代码获取；例如 `const aclTensor* x` 的 `type` 为 `aclTensor`。
     - 参数类型必须以当前函数原型和该参数说明中的基础类型为准，不要凭经验把 `float*`、`aclTensor*` 等泛化成其他类型。
     - **常见值**：可参考 ParameterType。
     - 参数类型只保留基础类型名称，**必须去掉 `*`、`const`、`struct` 等所有修饰符**，只保留干净类型名。例如：
       - `const aclTensor *x1` → `aclTensor`
       - `uint64_t *workspaceSize` → `uint64_t`
       - `aclOpExecutor **executor` → `aclOpExecutor`

   - **4.4 is_optional（是否可选）**：
     - 只有文档中出现以下任一明确信号时，才标记为 `true`：
       - 可选、Optional、default、可为空、可以为 nullptr、可不传、缺省值
     - “支持空Tensor”**不等于**参数可选；这只能说明参数值形态受限，不能据此把 `is_optional` 设为 `true`。
     - 参数名中包含 `Optional` 也不必然表示可选，只有文档明确说明可选或可为空时才标记为 `true`。
     - 其余情况一律为 `false`。

   - **4.5 description（参数描述）**：
     - 直接复用“描述”列 / “说明”列的完整内容。
     - 允许轻微精简冗长重复部分，但禁止添加、删减、改写核心含义。
     - 如果描述为空，填写：`"文档中未提供该参数说明"`
   - **4.6 format（数据格式）**：
     - 从 "数据格式" 列提取
     - 单个格式如 `ND` 使用字符串 `"ND"`，多个格式才使用数组。
     - 如果文档中数据格式显示"-"，format字段应为 JSON `null`，不要写成字符串 `"null"`。

## 边界情况处理

- 文档每个函数都有一个对应的参数表(参数说明)，优先提取主执行函数的参数，其他函数可省略或标注“同上”。
- 存在重载函数 → 直接使用第一个。
- 参数表缺失 → 尝试从函数原型 `(...)` 中解析，但需在 description 注明“从函数签名推断，缺少详细说明”。
- 无任何函数原型信息 → `functions: []` 并在主输出中说明“文档未提供函数原型信息”。
