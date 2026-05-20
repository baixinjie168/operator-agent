# platform_specifics 提取规则（平台特异性信息）

本文件定义了从 CANN 算子说明文档中提取 **platform_specifics（平台特异性信息）** 的严格规则，用于描述特定硬件平台（如 Atlas A2、Atlas A3）下的特殊约束或要求。

## 目标输出数据结构
```python 
class PlatformSpecific(BaseModel):
    """平台特定规则模型"""
    platform: str = Field(..., description="平台名称")
    description: str = Field(..., description="平台描述")
    constraint_detail: str = Field(..., description="约束详情表达式")

    model_config = {"extra": "forbid"}

class PlatformSpecificsOutput(BaseModel):
    """顶层模型"""
    platform_specifics: List["PlatformSpecific"] = Field(default_factory=list, description="平台特定规则")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

- `platform_specifics` 是一个数组，每个元素代表一个特定平台下的特殊规则。
- `platform` 字段记录适用的平台名称，必须与文档中的原文完全一致。
- `description` 字段是约束的完整自然语言描述，必须直接复用文档中的原文。
- `constraint_detail` 字段用于提供一个机器可解析的 Python 表达式；如果文档中没有稳定表达式，该字段使用空字符串 `""`，不要使用 `null`。

## 提取规则（逐字段）

1. **识别 platform_specifics 区域**
   - 优先级顺序查找以下关键词或结构（任一命中即可）：
     - 小节标题包含 `限制`、`约束`、`要求`、`特异性`、`特定平台`、`硬件要求`、`支持矩阵` 等。
     - 以 `注意`、`说明`、`限制` 开头，且明确提到某个平台的段落或列表项。
     - 文本中包含 `是否支持`、`仅支持`、`不支持`、`某平台下无效`、`某平台要求` 等表述。
   - 只要文档明确存在平台差异、支持矩阵或平台专属限制，就应考虑抽取到 `platform_specifics`。
   - 不要把单个参数的数据类型支持表误放进 `platform_specifics`；只有不同平台的 dtype 支持确实不同，才作为平台差异记录。

2. **platform（平台信息）**
   - 从约束描述的上下文中提取平台名，必须完整保留文档原文中的平台写法。
   - 如果同一条规则明确同时适用于多个平台，可以保留为一个组合平台字符串；如果不同平台规则不同，必须拆成多条记录。
   - 不要把与平台无关的说明、dtype 组合、自然语言解释写进 `platform`。

3. **description（约束描述）**
   - `description` 用于保留人类可读的约束说明，应准确表达该平台下的限制、差异或支持状态。
   - 可以复用原文句子或段落，但不要把多个互不相干的平台规则混成同一条描述。
   - 当一段文字里同时包含多条独立的平台要求时，应按语义拆分为多条记录，而不是堆成一段长描述。

4. **constraint_detail（机器可解析规则）**
   - `constraint_detail` 必须是字符串，用于承载 machine-readable 的约束表达式。
   - 对产品支持矩阵，`description` 应记录支持/不支持状态；`constraint_detail` 只在需要把支持状态或平台专属限制转成机器约束时填写，例如 `operator_supported == true` 或 `operator_supported == false`。如果没有稳定表达式，使用空字符串 `""`，不要把 `supported == true/false` 伪装成 dtype、shape 等参数约束。
   - 对于 dtype 限制、模式开关、`cubeMathType`、`divMode`、shape/字节对齐等平台条件，只要能稳定转写，就应提供简洁、可判定的表达式。
   - 只有当文档内容确实无法稳定抽象为表达式时，`constraint_detail` 才可以为空字符串 `""`；不要因为偷懒而留空。
   - `constraint_detail` 必须与 `description` 语义一致，不要把整段自然语言、JSON、字典或数组对象直接塞进去。
   - 如果同一平台下同时有“是否支持”和“dtype 不支持 BFLOAT16/COMPLEX64”等独立限制，应拆成独立记录或在语义上清晰分开，避免把平台支持状态和参数约束混成一条。

## 边界情况处理

- **无平台特异性信息**：如果文档中完全未提及任何平台差异，则 `platform_specifics` 应为 `[]`。
- **全局规则**：对所有平台都成立的输入输出约束、shape 关系、dtype 关系，不应放入 `platform_specifics`；应转入 `parameter_constraints`、`inter_parameter_constraints` 或 `dtype_map`。
- **支持矩阵与差异规则**：如果文档显式列出平台支持矩阵，需尽量补全所有明确列出的平台，尤其不要遗漏不支持的平台；相同支持状态可以合并，不同状态必须拆分。
- **规则拆分**：同一平台下的多条独立限制应拆成多条记录；多个平台共享同一条规则时才允许合并。
- **隐式约束**：如约束只是弱提示或无法稳定判定，`description` 中可以注明“（推断）”，但 `constraint_detail` 需谨慎填写，避免过度推断。
