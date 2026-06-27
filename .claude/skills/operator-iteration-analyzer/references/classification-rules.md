# 分类规则详解

本文档详述 `classify_failure.py` 中每种分类的判定细节。

## 分类总览

| 分类代码 | 显示名 | 含义 | 典型信号 |
|----------|--------|------|----------|
| `CONSTRAINT_WRONG` | 约束提取错误 | json_constraints 字段值与源文档不符 | dimensions.value 异常、dtype 不匹配源文档 |
| `CONSTRAINT_MISSING` | 约束缺失 | 源文档中存在但 json_constraints 没有 | params 列表缺失、constraints_in_parameters 为空 |
| `GENERATOR_CODE_BUG` | 生成代码 Bug | 约束正确但 case_builder 逻辑错误 | traceback 指向 generators/ 模块 |
| `LLM_PROMPT_GAP` | LLM 提示词遗漏 | 提示词未约束某些情况 | 反复在同一类算子上失败 |
| `EXECUTION_ENV_ERROR` | 执行环境错误 | ATK 运行/驱动/环境问题 | ssh 失败、CudaError、ASCEND 变量未设 |
| `CONSTRAINT_GENERATOR_BOTH` | 约束+生成双端问题 | 两端都可能有问题 | 难以单一归因 |

## 判定流程图

```
开始
  │
  ├─ constraint_status != success ?
  │     └─ YES → CONSTRAINT_MISSING（json_constraints 为空）
  │             或 CONSTRAINT_WRONG（有内容但不合规）
  │
  ├─ case_generation_status == failed ?
  │     ├─ 日志含 Python 异常 + 栈在 generators/ → GENERATOR_CODE_BUG
  │     ├─ 日志含 "json_constraints not found" → CONSTRAINT_WRONG（约束未生成）
  │     └─ 其他 → GENERATOR_CODE_BUG（保守）
  │
  └─ execution_status in {failed, partial} ?
        ├─ 日志含 ssh/cuda/环境错误 → EXECUTION_ENV_ERROR
        ├─ failure_reason 含 shape invalid + 约束 shape 看起来合规 → GENERATOR_CODE_BUG
        ├─ failure_reason 含 shape invalid + 约束 shape 异常 → CONSTRAINT_WRONG
        ├─ failure_reason 含 dtype mismatch + outputs dtype 为空 → CONSTRAINT_MISSING
        ├─ failure_reason 含 dtype mismatch + outputs dtype 正常 → CONSTRAINT_GENERATOR_BOTH
        ├─ failure_reason 含 unknown parameter + 源文档有该参数 → CONSTRAINT_MISSING
        ├─ failure_reason 含 unknown parameter + 源文档也无 → CONSTRAINT_WRONG
        └─ 其他 → UNKNOWN
```

## 各分类的详细判定

### CONSTRAINT_WRONG（约束提取错误）

**触发条件**：

1. `document_versions.json_constraints` 字段非空，但字段值与源文档明显不符
2. 特定维度错误：
   - `dimensions.value` 为 None 或非法格式
   - `dtype.value` 与源文档数据类型列不符
   - `is_support_discontinuous.value` 与源文档非连续 Tensor 列不一致

**判定逻辑**：

```python
# 在 analyze_constraint.py 中实现
_check_param_completeness(jc, src_doc)  # 参数完整性
_check_shape_consistency(jc)            # shape 格式
_check_constraints_in_params(jc, src_doc)  # relation 完整性
```

**典型场景**：
- LLM 把 `dimensions.value` 输出为 `null`
- LLM 把 Tensor 的 `dtype.value` 输出为 `[]`
- LLM 漏写了 `constraints_in_parameters`

### CONSTRAINT_MISSING（约束缺失）

**触发条件**：

1. 源文档参数表或加粗文本中存在参数名 X，但 `json_constraints.inputs/outputs` 没有 X
2. `constraints_in_parameters` 完全为空
3. 源文档使用说明列中明确提到某种约束，但 JSON 中没有对应 expr

**判定逻辑**：

```python
doc_params = set(re.findall(r"\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*", src_doc))
json_params = set((jc.get("inputs") or {}).keys()) | set((jc.get("outputs") or {}).keys())
missing = doc_params - json_params
if missing:
    → CONSTRAINT_MISSING
```

**典型场景**：
- 隐式参数（不在函数签名中的 alpha、axis）被 implicit_param_extract 漏掉
- 衍生参数（如 batch_size、hidden_size）被遗漏
- 参数间约束（如"X 与 Y 同类型"）未生成 expr

### GENERATOR_CODE_BUG（生成代码 Bug）

**触发条件**：

1. 日志中存在 Python traceback，且栈帧在 `packages/agent/src/agent/generators/` 路径下
2. 约束看起来合规，但生成的 case 无法通过 ATK（典型：shape/dtype 越界）
3. 异常类型多为：`AttributeError`、`KeyError`、`IndexError`、`TypeError`、`ValueError`

**判定逻辑**：

```python
if re.search(r"File \".*generators/(case_builder|shape_sampler|value_sampler|dtype_picker|shape_groups)\.py\"", log_text):
    → GENERATOR_CODE_BUG
```

**典型场景**：
- `dtype_picker` 在 `param.dtype` 为空列表时返回 None，后续 `.lower()` 失败
- `shape_sampler` 在 `dimensions.value` 为 None 时崩溃
- `case_builder` 的 `idx` 越界
- `value_sampler` 的范围检查 bug

### LLM_PROMPT_GAP（LLM 提示词遗漏）

**触发条件**：

1. 同一类约束问题在多个算子上反复出现（≥ 3 个算子同样的 field_name 出错）
2. 约束问题无法用单一算子的源文档差异解释 → 说明是 prompt 通用问题
3. `secondary_categories` 中标注了 `LLM_PROMPT_GAP`

**判定逻辑**（在 `analyze_prompt.py` 中基于关键词匹配）：

```python
keywords_to_prompts = {
    "dimensions.value": "SHAPE_EXTRACT_PROMPT",
    "dtype.value": "DTYPE_EXTRACT_PROMPT",
    "param_missing": "IMPLICIT_PARAM_EXTRACT_PROMPT",
    "constraints_in_parameters 为空": "PARAM_RELATION_EXTRACT_PROMPT",
    ...
}
```

**典型场景**：
- 所有算子的 alpha 参数都漏提 → IMPLICIT_PARAM_EXTRACT_PROMPT 缺少 alpha 示例
- 所有算子的 bidirectional 标志都漏提 → PARAM_RELATION_EXTRACT_PROMPT 缺少 RNN 示例

### EXECUTION_ENV_ERROR（执行环境错误）

**触发条件**：

1. 日志/failure_reason 包含以下任一关键词：
   - `ssh: connect to host.*port.*Connection refused`
   - `No such file or directory:.*atk`
   - `ASCEND_RT_ERROR`
   - `RuntimeError.*device|driver`
   - `CudaError`
   - `timeout.*expired`

**典型场景**：
- 远程执行机 SSH 不通
- ATK 工具未安装
- NPU 驱动未加载
- CANN 环境变量未设置

### CONSTRAINT_GENERATOR_BOTH（双端问题）

**触发条件**：

1. 失败原因同时涉及约束和生成两端，无法确定单一根因
2. 置信度 ≤ 0.6

**典型场景**：
- failure_reason 同时提到 dtype 错误，但约束和生成代码都可能有问题
- 生成的 shape 在约束范围内，但 ATK 不接受

## 置信度计算

| 情况 | 置信度 |
|------|--------|
| 约束缺失 + 多个缺失证据 | 0.9 |
| 生成代码 traceback + generators/ 路径 | 0.85 |
| 约束异常 + 至少 1 处具体字段错误 | 0.85 |
| 失败原因明确指向执行环境 | 0.9 |
| 仅有日志但无 traceback | 0.6 |
| 失败原因模糊、未匹配任何模式 | 0.3 |

## 输出示例

```json
{
    "operator_name": "aclnnAbs",
    "category": "GENERATOR_CODE_BUG",
    "confidence": 0.85,
    "evidence": [
        "AttributeError: 'NoneType' object has no attribute 'lower'",
        "异常发生在 generators/ 模块"
    ],
    "secondary_categories": [],
    "notes": "用例生成日志出现 Python 异常，且堆栈指向 generators/ 模块"
}
```

## 如何改进分类准确率

1. **扩展 `_GENERATOR_CODE_PATTERNS`**：根据历史失败日志添加新 pattern
2. **增加约束检查规则**：在 `analyze_constraint.py` 中添加新检测
3. **支持更多 prompt 映射**：在 `_FAILURE_TO_PROMPTS` 中添加新的失败模式
4. **累积学习**：将多次分析结果汇总，统计高频问题模式