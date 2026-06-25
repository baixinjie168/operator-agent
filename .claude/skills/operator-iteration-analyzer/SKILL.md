---
name: operator-iteration-analyzer
description: 算子执行结果驱动的 operator-agent 项目迭代分析优化工具。自动扫描 operator-agent 项目运行数据（DB 中 document_versions.json_constraints、batch_cases/{operator}_cases.json、execution_results/{operator}/result.json、logs/generate_case_{operator}.log），定位算子在约束提取、用例生成、用例执行各阶段的失败根因（约束错误 / 生成代码错误 / LLM 遗漏 / 执行逻辑错误），输出带修复建议的结构化分析报告（Markdown + HTML 双格式）。支持单算子深度分析和全量算子批量分析两种模式。当用户提到"分析算子失败原因"、"定位约束提取错误"、"分析执行结果失败"、"迭代优化 operator-agent"、"分析算子用例失败"、"检查算子端到端问题"、"operator-agent 调优"等场景时使用此 skill。即使用户只是说"分析一下这个算子为什么失败"或"哪些算子出问题了"，也应该触发此 skill。
---

# 算子执行结果驱动的 operator-agent 迭代分析

基于 operator-agent 项目运行产生的真实数据（数据库 + 日志 + 产物文件），定位算子在"约束提取 → 用例生成 → 用例执行"全链路上的失败根因，输出可执行的修复建议。

## 触发格式

用户提供以下任一格式触发此 skill：

```
使用operator-iteration-analyzer skill
模式：单算子 | 全量
算子名称：<operator_name>      # 单算子模式必填；全量模式可省略
```

**模式说明**：
- **单算子模式**：分析指定算子的全链路状态。深度分析约束、用例、执行三阶段，给出精确修复建议
- **全量模式**：扫描所有成功提取约束的算子，统计各阶段成功率，列出 TopN 失败算子，输出汇总报告

**默认行为**（用户未指定算子时）：使用全量模式。

## 数据源

skill 默认读取以下路径（均相对于 operator-agent 项目根目录）：

| 数据源 | 路径 | 用途 |
|--------|------|------|
| 约束数据库 | `data/operator_agent.db`（document_versions.json_constraints） | 判断算子是否成功提取约束 |
| 用例产物 | `batch_cases/{operator_name}_cases.json` | 判断用例是否生成成功 |
| 用例产物（按产品） | `cases/{operator}_{product_safe}_cases.json` | case_subgraph 产物的兼容路径 |
| 执行结果 | `execution_results/{operator_name}/result.json` | 判断用例执行结果 |
| 生成日志 | `logs/generate_case_{operator_name}.log` | 提取生成失败原因 |
| 算子源文档 | `operators/{operator_name}.md` | 用于人工校对约束提取错误 |
| 提示词目录 | `packages/agent/src/agent/prompts/` | 定位需要优化的提示词 |
| 节点代码 | `packages/agent/src/agent/nodes/` | 定位需要修复的代码 |

如果用户提供了不同的项目根路径，使用用户提供的路径覆盖默认值。

## 工作流程

### Step 1: 收集所有算子状态

**目标**：列出所有"成功提取约束"的算子，并标注每个算子在各阶段的状态。

调用 [`scripts/scan_operators.py`](scripts/scan_operators.py) 脚本完成：

```bash
python .claude/skills/operator-iteration-analyzer/scripts/scan_operators.py \
    --project-root <project_root> \
    [--operator <operator_name>] \
    --output scan_results.json
```

脚本输出结构：

```json
{
  "operators": [
    {
      "operator_name": "aclnnAbs",
      "doc_id": 1,
      "constraint_status": "success",
      "constraint_keys_count": 8,
      "case_generation_status": "success | failed | missing",
      "case_files": ["batch_cases/aclnnAbs_cases.json"],
      "execution_status": "success | failed | partial | missing",
      "execution_dir": "execution_results/aclnnAbs",
      "result_json_path": "execution_results/aclnnAbs/result.json",
      "report_records": [
        {"id": 0, "run_result": "SUCCESS", "failure_reason": null},
        {"id": 1, "run_result": "FAILED", "failure_reason": "..."}
      ],
      "log_path": "logs/generate_case_aclnnAbs.log"
    }
  ],
  "summary": {
    "total": 10,
    "constraint_success": 10,
    "case_generation_success": 8,
    "case_generation_failed": 2,
    "execution_success": 5,
    "execution_failed": 3
  }
}
```

每个算子的状态分四层：

1. **constraint_status**：`success`（有非空 json_constraints）/ `failed`（json_constraints 为空或 "{}"）/ `missing`（无 document_versions 记录）
2. **case_generation_status**：`success`（存在 *_cases.json）/ `failed`（日志中有异常堆栈或 Python 错误）/ `missing`（无产物且无日志）
3. **execution_status**：`success`（report_records 全部 SUCCESS）/ `failed`（存在 FAILED）/ `partial`（部分成功）/ `missing`（无 result.json）
4. **report_records**：从 result.json.task_report_data.report_records 提取

### Step 2: 对每个失败算子进行根因分类

调用 [`scripts/classify_failure.py`](scripts/classify_failure.py) 脚本完成根因分类。

**输入**：scan_results.json 中的单个失败算子 + 失败原因（生成日志末尾 / execution failure_reason）

**输出分类**（按优先级）：

| 分类代码 | 类别 | 含义 | 后续动作 |
|----------|------|------|----------|
| `CONSTRAINT_WRONG` | 约束不正确 | json_constraints 中有字段与源文档 Markdown 不一致 | 跳到 Step 3 |
| `CONSTRAINT_MISSING` | 约束缺失 | 源文档中有但 json_constraints 中没有 | 跳到 Step 3 |
| `GENERATOR_CODE_BUG` | 生成代码错误 | constraints 正确，但 generator 逻辑错误导致非法 case | 跳到 Step 4 |
| `LLM_PROMPT_GAP` | LLM 提示词遗漏 | 约束被 LLM 漏掉，需要优化提示词 | 跳到 Step 5 |
| `EXECUTION_ENV_ERROR` | 执行环境错误 | ATK 运行时环境问题（设备/驱动/路径） | 跳到 Step 6 |
| `CONSTRAINT_GENERATOR_BOTH` | 双端都可能有问题 | 约束和生成都疑似有问题 | 同时执行 Step 3 + Step 4 |

**分类判定逻辑**（脚本中实现）：

```
1. 读取算子源文档 operators/{name}.md
2. 读取 document_versions.json_constraints
3. 读取生成日志（或 execution failure_reason）
4. 对照判定：
   a. 若日志中提到 "json_constraints not found" → 约束未生成，跳到 Step 3
   b. 若日志中提到 KeyError/AttributeError 等 Python 异常，且异常栈指向 generators/case_builder.py
      → GENERATOR_CODE_BUG（约束正确，是生成代码 bug）
   c. 若 execution failure_reason 中提到 shape/dtype 不匹配 ATK 规范，且 json_constraints 中相关字段看起来合规
      → GENERATOR_CODE_BUG（generator 选错了 shape/dtype）
   d. 若 execution failure_reason 中提到 ATK API 错误（如 "tensor shape invalid"），且 json_constraints 中对应参数 dimensions.value 与源文档 shape 列不符
      → CONSTRAINT_WRONG（约束提取错了 dimensions）
   e. 若 json_constraints 中 inputs/outputs 缺少源文档中的参数
      → CONSTRAINT_MISSING 或 LLM_PROMPT_GAP
   f. 若源文档中明确描述了约束，但 json_constraints 完全没体现
      → LLM_PROMPT_GAP
   g. 若 ATK 报错信息与 operator 无关（如设备未连接、CUDA 错误）
      → EXECUTION_ENV_ERROR
```

### Step 3: 约束错误 / 缺失深度分析

调用 [`scripts/analyze_constraint.py`](scripts/analyze_constraint.py)。

**输入**：算子源文档 + json_constraints + 分类结果

**输出**：每个有问题的字段的详细分析：

```json
{
  "operator_name": "aclnnAbs",
  "constraint_issues": [
    {
      "category": "param_missing",
      "param_name": "alpha",
      "src_doc_evidence": "参数表第3行：alpha | 输入 | 缩放系数",
      "json_constraints_state": "inputs 中不存在 alpha",
      "likely_root_cause": "LLM 在 implicit_param_extract 阶段漏掉非函数签名参数",
      "affected_prompt": "packages/agent/src/agent/prompts/system.py 或节点 llm_description_extract 子图",
      "affected_code": "packages/agent/src/agent/nodes/implicit_param_extract.py",
      "fix_suggestion": "在 implicit_param_extract 的 prompt 中加入示例：'对于不在函数签名中但参数表中出现的参数（如 alpha、axis），请显式提取并加入 inputs'"
    }
  ]
}
```

### Step 4: 生成代码错误定位

调用 [`scripts/analyze_generator.py`](scripts/analyze_generator.py)。

**输入**：json_constraints + 失败用例 + 生成日志异常堆栈 + generators 目录代码

**输出**：具体代码位置和修复建议：

```json
{
  "operator_name": "aclnnAbs",
  "generator_issues": [
    {
      "file": "packages/agent/src/agent/generators/case_builder.py",
      "function": "build_single_case",
      "line": 145,
      "code_snippet": "dtype = pick_dtype_for_param(rng, param, platform)",
      "issue": "当 param.dtype 为空列表时，pick_dtype_for_param 返回 None，导致后续 .lower() 失败",
      "exception": "AttributeError: 'NoneType' object has no attribute 'lower'",
      "fix_suggestion": "在 pick_dtype_for_param 中添加空列表兜底，返回 'FLOAT32' 或随机选一个支持的 dtype"
    }
  ]
}
```

### Step 5: LLM 提示词优化建议

调用 [`scripts/analyze_prompt.py`](scripts/analyze_prompt.py)。

**输入**：约束错误模式 + 相关节点的 prompt 文件

**输出**：具体的提示词优化建议：

```json
{
  "operator_name": "aclnnAbs",
  "prompt_improvements": [
    {
      "prompt_file": "packages/agent/src/agent/prompts/system.py",
      "prompt_name": "DTYPE_EXTRACT_PROMPT",
      "current_excerpt": "...(原 prompt 文本)",
      "improvement": "增加 dtype 互斥示例：当输入是 INT8 时，输出不能是 FLOAT32",
      "rationale": "本次失败是因为 json_constraints 中 input dtype 包含 INT8，但 output dtype 列表为空，ATK 报 dtype mismatch",
      "added_example": "### Example 3\ninput: x 是 INT8\noutput: y 必须与 x 同类型 → y.dtype.value = ['INT8']"
    }
  ]
}
```

### Step 6: 执行环境错误处理

输出标准的环境排查清单：

```markdown
## 执行环境排查清单

- [ ] 检查 atk_executor_path 是否正确
- [ ] 检查远程执行机是否可达
- [ ] 检查 ATK 版本是否与算子匹配
- [ ] 检查设备驱动是否加载
- [ ] 收集 ATK 完整日志
```

### Step 7: 生成报告

调用 [`scripts/generate_report.py`](scripts/generate_report.py) 生成最终报告。

**支持的格式**：
- Markdown（默认）：`reports/{operator_name}_analysis.md` 或 `reports/all_operators_analysis.md`
- HTML（可选，需 --format html）：明亮风格，含可导航的目录

**报告结构**（Markdown）：

```markdown
# operator-agent 算子迭代分析报告

生成时间：2026-06-22 03:45:00
分析模式：单算子 | 全量
扫描算子数：10

## 一、整体概览

| 阶段 | 成功 | 失败 | 成功率 |
|------|------|------|--------|
| 约束提取 | 10 | 0 | 100% |
| 用例生成 | 8 | 2 | 80% |
| 用例执行 | 5 | 3 | 50% |

## 二、失败算子明细

### 2.1 aclnnAbs
- 分类结果：GENERATOR_CODE_BUG
- 失败用例：id=1, run_result=FAILED
- 失败原因：AttributeError: 'NoneType' object has no attribute 'lower'

**问题代码**：
`packages/agent/src/agent/generators/case_builder.py:145`
```python
dtype = pick_dtype_for_param(rng, param, platform)
```

**根因**：当 param.dtype 为空列表时，pick_dtype_for_param 返回 None

**修复建议**：
```python
dtype = pick_dtype_for_param(rng, param, platform)
if dtype is None:
    dtype = "FLOAT32"  # 兜底
```

## 三、TopN 失败算子优先级

1. **aclnnAbs** (3 个失败用例) — GENERATOR_CODE_BUG
2. **aclnnAmin** (2 个失败用例) — CONSTRAINT_WRONG

## 四、迭代优化建议

1. [ ] 修复 case_builder.py:145 的 dtype 兜底逻辑
2. [ ] 优化 DTYPE_EXTRACT_PROMPT，添加 dtype 互斥示例
3. [ ] 在 implicit_param_extract 节点增强非签名参数提取
```

## 报告输出路径

- 单算子模式：`reports/{operator_name}_analysis.md`（默认在项目根目录）
- 全量模式：`reports/all_operators_analysis.md`

用户可通过 `--output` 参数指定输出路径。

## 输出摘要

最终向用户汇报：

1. **整体成功率统计**：约束/用例/执行三阶段成功率
2. **失败算子 TopN**：按失败严重程度排序
3. **分类汇总**：CONSTRAINT_WRONG / GENERATOR_CODE_BUG / LLM_PROMPT_GAP 各占多少
4. **可执行修复清单**：每个失败的修复建议
5. **报告文件路径**

## 注意事项

1. **不要自动修改代码**：本 skill 只做分析和出报告，所有修复建议需要用户确认后再执行
2. **数据安全**：分析时只读不写，不要修改 DB、cases、execution_results 等产物
3. **大文件处理**：execution_results/result.json 可能很大，只读取 task_report_data.report_records 字段
4. **多产品维度**：json_constraints 按产品分组，分析时需要针对每个产品分别判断
5. **cases 路径兼容**：case_subgraph 节点产物路径是 `cases/{op}_{product_safe}_cases.json`，旧的 generate_cases 节点产物是 `cases/{op}_cases.json` 或 `batch_cases/{op}_cases.json`，需要同时扫描三种路径
6. **缺失文件降级**：如果 logs/generate_case_*.log 不存在，不要报错，改用 pipeline_runs 表中的 error 字段
7. **JSON 解析容错**：json_constraints 字段可能损坏，解析失败时记录到 warning 而非崩溃

## 脚本索引

| 脚本 | 用途 |
|------|------|
| [scan_operators.py](scripts/scan_operators.py) | 扫描所有算子状态 |
| [classify_failure.py](scripts/classify_failure.py) | 根因分类 |
| [analyze_constraint.py](scripts/analyze_constraint.py) | 约束错误深度分析 |
| [analyze_generator.py](scripts/analyze_generator.py) | 生成代码错误定位 |
| [analyze_prompt.py](scripts/analyze_prompt.py) | 提示词优化建议 |
| [generate_report.py](scripts/generate_report.py) | 生成最终报告 |

## 相关文档

- [数据源说明](references/data-sources.md) — 各数据源的字段映射
- [分类规则详解](references/classification-rules.md) — 根因分类的判定细节
- [提示词优化模式](references/prompt-patterns.md) — 常见 LLM 提示词问题的修复模式