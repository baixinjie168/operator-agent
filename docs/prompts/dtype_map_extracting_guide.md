# dtype_map 提取规则

本文件定义了从 CANN 算子说明文档 中提取 **dtype_map（数据类型支持）** 的严格规则。

## 目标输出数据结构
```python 
class DtypeMapItem(BaseModel):
    """数据类型映射表项模型"""
    platform: str = Field(..., description="平台名称")
    columns: List[str] = Field(..., description="列名列表")
    rows: List[List[str]] = Field(..., description="数据类型组合行列表")

    model_config = {"extra": "forbid"}

class DTypeMapOutput(BaseModel):
    """顶层模型"""
    dtype_map: List["DtypeMapItem"] = Field(default_factory=list, description="数据类型映射表")

    model_config = {"extra": "forbid"}
```
【输出规则（必须严格遵守，缺一不可）】
1.  输出格式：仅返回纯JSON字符串，无任何多余内容（无解释、无代码块、无换行备注）；
2.  输出范围：只返回【顶层类】的完整结构，自动嵌套填充所有内层类，不单独输出任何内层类（如内层类1、内层类2）；
3.  字段约束：字段名、字段类型、层级结构，必须与上面定义的类完全一致，禁止新增、缺失、修改字段；
4.  类型匹配：严格遵循类型注解（str/int/bool/List/Optional等），空值统一用null（JSON规范），不随意填充无效值；
5.  嵌套要求：所有嵌套结构必须完整，若待处理内容中无相关信息，可选字段填默认值，必填字段（Field(...)）填合理空值（如""、0、[]）。

- `dtype_map` 是一个数组，每个元素代表一个硬件平台（如 Atlas A2、Atlas A3）的组合支持情况。
- `platform` 字段记录平台名称。
- `columns` 数组按顺序列出所有输入/输出参数的名称（不含“数据类型”后缀）。
- `rows` 数组是一个二维数组，每行代表一种完整的数据类型组合，顺序与 `columns` 一致。

## 提取规则
  - 扫描算子文档中的 **约束说明**、**各产品支持数据类型说明**、**数据类型组合表**、**支持矩阵** 等区域，确认是否存在“显式的 dtype 映射/组合表”。
  - 只有文档明确给出 **多参数 dtype 组合表、输入输出 dtype 对应表、量化映射表或显式类型映射关系** 时，才生成 `dtype_map`。
  - 如果文档只是描述单个参数支持哪些 dtype，或描述“某参数与另一参数 dtype 一致”，不要生成 `dtype_map`；这类信息应分别落到 `parameter_constraints.data_types` 或 `inter_parameter_constraints`。
  - 类似“输入为 INT/BOOL 时输出为 FLOAT”这类文字性跨参数关系，不属于 `dtype_map`，应放到 `parameter_constraints` 的 dtype 枚举和 `inter_parameter_constraints` 的类型依赖中。
  - 如果存在数据类型组合表格：
      - 识别该表格适用的平台（例如“Atlas A2 ...”），并在 `dtype_map` 中创建对应条目；不同平台表格内容不同必须拆分为多个条目。
      - 提取表格标题，并将其映射到函数中定义的确切参数名称（例如“x1数据类型” -> “x1”）；`columns` 中只保留真实参数名，不保留“数据类型”后缀，也不要把备注列、平台列、序号列写进去。
      - 将每一行有效的类型组合写入 `rows`；每个 dtype 值都必须忠实保留为文档中的合法类型名称，通常为大写字符串（例如“FLOAT32”）。
      - 不要凭空补全表中未出现的组合，也不要把表头、注释行、空行、自然语言说明写进 `rows`。
  - 如果文档不存在显式的数据类型映射/组合表：
      - 将 `dtype_map` 字段设置为空列表：`[]`。
  - 若存在平台差异的 dtype 映射，必须按平台拆分；不要把多平台差异错误合并为一条全局记录。
  - 坚持“无显式映射，不生成 `dtype_map`”的保守策略，不要从支持类型列表、经验规则或其他模块约束中反推 `dtype_map`。
