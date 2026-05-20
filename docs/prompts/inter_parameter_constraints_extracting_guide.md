# inter_parameter_constraints 提取规则

本文件定义了从 CANN 算子说明文档 中提取 **inter_parameter_constraints（参数之间的约束关系）** 的严格规则。

## 目标输出数据结构
```python 
class InterConstraintsRuleType(str, Enum):
    """参数间约束类型枚举"""
    SHAPE_BROADCAST = "shape_broadcast"
    SHAPE_CHOICE = "shape_choice"
    SHAPE_EQUALITY = "shape_equality"
    SHAPE_DEPENDENCY = "shape_dependency"
    SHAPE_VALUE_DEPENDENCY = "shape_value_dependency"
    PRESENCE_DEPENDENCY = "presence_dependency"
    TYPE_DEPENDENCY = "type_dependency"
    TYPE_EQUALITY = "type_equality"
    VALUE_DEPENDENCY = "value_dependency"
    FORMAT_EQUALITY = "format_equality"

class InterParamConstraint(BaseModel):
    """参数间约束模型"""
    type: InterConstraintsRuleType = Field(..., description="约束类型")
    params: List[str] = Field(..., description="涉及的参数列表")
    expr: str = Field(..., description="约束表达式")
    description: str = Field(default="", description="约束描述")
    source: Optional[str] = Field(default=None, description="源参数")
    target: Optional[List[str]] = Field(default=None, description="目标参数列表")

    model_config = {"extra": "forbid"}

class InterParameterConstraintsOutput(BaseModel):
    """顶层模型"""
    inter_parameter_constraints: List["InterParamConstraint"] = Field(default_factory=list, description="参数间约束")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

## 函数属性

### type
`type`属性用于指定参数之间的约束类型。该值必须是以下字符串字面量之一：
- `shape_broadcast`：表示形状需要满足广播（broadcast）关系。
- `shape_choice`：表示一个形状可以是多个候选参数之一。
- `shape_equality`：表示形状必须完全相同。
- `shape_dependency`：表示 shape 元素值之间的依赖关系。
- `shape_value_dependency`：表示特定轴值之间的依赖关系。
- `value_dependency`：表示张量元素值或参数值之间的约束关系。
- `presence_dependency`：表示共存规则（例如 `(A is None) == (B is None)`）。
- `type_dependency`：表示数据类型依赖关系。
- `type_equality`：表示数据类型必须一致。
- `format_equality`：表示数据格式必须一致。
该值必须严格遵循上述选项之一，不允许使用其他值。
- 只抽取真正依赖多个参数的约束；单参数约束、公式解释或功能性描述不要放进 `inter_parameter_constraints`。
- 先判断语义再选 `type`，不要把 dependency、choice、equality、value_dependency 等关系混用。
- 像“shape 为 [B,H] 或 [B,1,H]，且由 x 推导”这类关系，本质上是 `shape_dependency`，不是 `shape_choice`。
- 具体轴值关系、拼接轴求和、非拼接轴相等、输出 rank 与输入 rank 一致等，应优先使用 `shape_value_dependency`；整体 shape 来源或广播关系才使用 `shape_dependency` / `shape_broadcast`。
- “与另一参数同 dtype / 同 format / 同 shape”这类显式跨参数关系，应优先在本模块中选择语义最准确的 `type` 表达。
- dtype 推导、可转换、同 dtype 等跨参数关系应在本模块用 `type_dependency` 或 `type_equality` 表达，不要塞进 `parameter_constraints.data_types.types`。

### params
`params`属性是一个字符串列表，用于指定约束所涉及的参数。约束表达式中提到的所有参数都必须包含在此列表中。
- `expr`字段中引用的所有参数都必须包含在`params`列表中。
- `source` 和 `target` 中出现的参数也必须包含在 `params` 列表中。
- 不要包含与约束无关的参数。
- 列表中应仅包含参数名称作为字符串。
- 不要漏掉共同参与推导的参数，例如 `pad`、`dim`、`tensors` 等。
  
### expr
`expr`：必填字段（CRITICAL）。你必须为每个参数间约束提供一个正式、合法的 Python 表达式，以表示逻辑关系，禁止仅使用标点符号表示逻辑。
`expr`字段的规则：
  - `expr` 不允许为 `null`；如果当前约束暂时无法写出表达式，也必须使用空字符串 `""`，不要输出 `null`。
  - 对 `type_dependency`、`shape_dependency`、`value_dependency` 等依赖类约束，如果无法写出可判定表达式，应优先不输出该条约束；不要保留一条空表达式的依赖关系。
  - 不要使用 `implies` 这个词。
  - 不要使用伪代码。
  - 不要在 `expr` 中使用平台值作为判断条件。
  - 不要使用自然语言式 `if/else` 说明语句。
  - 蕴含逻辑：对于“如果 A 则 B”的逻辑，优先使用合法 Python 布尔表达式，例如 `(B) if (A) else True`，或其他语义等价且可执行的布尔表达式。
  - 等价逻辑：对于“A当且仅当B”的逻辑，使用相等性 `(A) == (B)`；例如 `(scales2 is None) == (zeroPoints2 is None)`。
  - 文档资料中的 tensor “元素个数”是指 tensor 中所有元素数量；若 shape 为 `[x, y, z]`，则元素个数为 `x * y * z`。
  - Python 表达式必须合法有效，且返回值为 bool；若涉及生成器表达式，必须包裹在 `all()` 或 `any()` 等函数中，且不允许使用 `lambda`。
  - `expr` 必须只引用本条约束真正涉及的参数；`params` 要覆盖 `expr` 中出现的关键参数。
  - 除非是在表达合法的蕴含逻辑，否则不要滥用三元表达式。

### description
`description`属性提供约束的人类可读描述。应以自然语言清晰解释该约束。
  - 应以清晰、简洁的语言描述约束。
  - 应引用所涉及的具体参数。
  - 应解释约束的性质（例如，“输入x1、x2和输出yOut、xOut必须具有相同的形状”）。
  - 如果没有描述，该字段可以为空字符串。

### source
`source`属性用于指定方向性约束关系中的源参数。
  - 当参数之间存在明显方向性的关系时使用。
  - 例如，在 broadcast、choice、dependency 等关系中，源参数通常是“作为基准”的那个参数。
  - 该字段是可选的；如果关系不是方向性的，则可以省略。
  - 当存在时，应包含一个参数名称作为字符串。
  - 所有 dependency 类约束都应优先补齐 `source`，不要把明显有方向的关系留成 `null`。
  - 如果 `out.shape` 由输入和 `pad` / `dim` 推导，`source` 应选择主要来源参数，并在 `params` 中保留其他共同参与推导的参数。

### target
`target`属性用于指定方向性约束关系中的目标参数。
  - 当参数之间存在方向性关系时使用。
  - 例如，在 broadcast、choice、dependency 等关系中，目标参数是被约束或被推导的参数。
  - 该字段是可选的；如果关系不是方向性的，则可以省略。
  - 当存在时，应包含参数名称的字符串列表。
  - 当多个参数受同一源参数影响时，可以包含多个目标参数。
  - 所有 dependency 类约束都应优先补齐 `target`，体现“谁约束谁、谁依赖谁”的方向。
  - 如果被推导或被约束的是输出参数，`target` 至少应包含该输出参数，例如 `["out"]`。

## 重要提醒
  - expr 表达式应尽可能简洁，不包含冗余条件。
  - 所有表达式字段都要输出字符串；没有内容时使用空字符串 `""`，不要使用 `null`。
  - 如果文档关系需要额外中间符号，先确认这些符号是否应由 `other_parameters` 定义，再在本模块的 `expr` 中引用。
  - 不要把冗余关系、重复关系或本应删除的单参数限制硬塞进 `inter_parameter_constraints`。
