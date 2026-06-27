# operator-iteration-analyzer Skill

> 算子执行结果驱动的 operator-agent 项目迭代分析优化工具

## 快速开始

### 触发方式

**自然语言触发**（推荐）：

```
使用operator-iteration-analyzer skill分析所有算子
使用operator-iteration-analyzer skill分析算子 aclnnAbs
分析 operator-agent 项目的失败算子
哪些算子出问题了？
分析算子 aclnnAdaLayerNorm 为什么失败
```

### 命令行调用

```bash
# 全量分析
python .claude/skills/operator-iteration-analyzer/scripts/run_analysis.py \
    --project-root /path/to/operator-agent \
    --output-dir reports/iteration_analysis \
    --format both

# 单算子分析
python .claude/skills/operator-iteration-analyzer/scripts/run_analysis.py \
    --project-root /path/to/operator-agent \
    --operator aclnnAbs \
    --output-dir reports/iteration_analysis \
    --format html
```

输出：
- `reports/iteration_analysis/all_operators_analysis.md` — 主报告
- `reports/iteration_analysis/scan_result.json` — 算子状态
- `reports/iteration_analysis/classification.json` — 根因分类
- `reports/iteration_analysis/constraint_analysis.json` — 约束分析
- `reports/iteration_analysis/generator_analysis.json` — 生成代码分析
- `reports/iteration_analysis/prompt_analysis.json` — 提示词建议

## 核心能力

### 1. 三阶段成功率统计

| 阶段 | 数据源 | 统计指标 |
|------|--------|----------|
| 约束提取 | `document_versions.json_constraints` | 非空率 |
| 用例生成 | `batch_cases/*.json` / `cases/*.json` | 产物存在率 |
| 用例执行 | `execution_results/{op}/result.json` | SUCCESS 比例 |

### 2. 根因分类

对每个失败算子，自动判断属于哪一类：

| 分类 | 显示名 | 典型表现 |
|------|--------|----------|
| `CONSTRAINT_WRONG` | 约束提取错误 | dimensions/dtype 字段异常 |
| `CONSTRAINT_MISSING` | 约束缺失 | 参数表有但 JSON 无 |
| `GENERATOR_CODE_BUG` | 生成代码 Bug | traceback 在 generators/ |
| `LLM_PROMPT_GAP` | LLM 提示词遗漏 | 同类问题反复出现 |
| `EXECUTION_ENV_ERROR` | 执行环境错误 | SSH/CudaError |
| `CONSTRAINT_GENERATOR_BOTH` | 双端问题 | 难以单一归因 |

### 3. 具体修复建议

每种分类都给出：
- **出错文件路径 + 行号**（生成代码 bug）
- **受影响 prompt 文件 + 修复模板**（提示词问题）
- **字段级对比**（约束错误）

## 使用场景

### 场景 1：每周一次全量巡检

```bash
# 1. 跑全量分析
python run_analysis.py --project-root . --output-dir reports/weekly

# 2. 查看报告
cat reports/weekly/all_operators_analysis.md

# 3. 对照可执行清单
grep "^- \[ \]" reports/weekly/all_operators_analysis.md
```

### 场景 2：某个算子失败需要 debug

```bash
# 1. 单算子深度分析
python run_analysis.py --project-root . --operator aclnnAbs \
    --output-dir reports/debug_aclnnAbs --format html

# 2. 浏览器打开 HTML 报告查看详情
```

### 场景 3：修复 prompt 后验证

```bash
# 1. 修改 prompt 后跑几个历史上失败的算子
python run_analysis.py --project-root . --operator aclnnAmin \
    --output-dir reports/verify_min

# 2. 对比前后报告，看分类是否从 CONSTRAINT_MISSING 变为 SUCCESS
```

## 输出报告示例

```markdown
# operator-agent 算子迭代分析报告

生成时间：2026-06-22 04:30:00
扫描算子数：10

## 一、整体概览

| 阶段 | 成功 | 失败 | 缺失/部分 | 成功率 |
|------|------|------|----------|--------|
| 约束提取 | 10 | 0 | — | 100% |
| 用例生成 | 8 | 2 | 0 | 80% |
| 用例执行 | 5 | 3 | 2 | 50% |

## 二、失败算子明细

### 2.3 GENERATOR_CODE_BUG（2 个算子）

#### 1. aclnnAbs
- 分类：生成代码 Bug（置信度 85%）
- 失败用例：id=1, 失败原因="shape mismatch: ..."
- 说明：约束正确，但 case_builder 在 param.dtype 为空列表时返回 None

#### 2. aclnnAmin
- 分类：生成代码 Bug（置信度 85%）
- ...

## 六、迭代优化优先级清单

### 6.2 可执行修复清单
- [ ] 修复算子 aclnnAbs 的生成代码 Bug
- [ ] 修复算子 aclnnAmin 的生成代码 Bug
```

## 文件结构

```
.claude/skills/operator-iteration-analyzer/
├── SKILL.md                    # 主入口（Claude 读这个）
├── README.md                   # 本文件（人读）
├── scripts/
│   ├── run_analysis.py         # 一键运行全流程
│   ├── scan_operators.py       # Step 1：扫描状态
│   ├── classify_failure.py     # Step 2：根因分类
│   ├── analyze_constraint.py   # Step 3：约束深度分析
│   ├── analyze_generator.py    # Step 4：生成代码定位
│   ├── analyze_prompt.py       # Step 5：提示词建议
│   └── generate_report.py      # Step 6：生成报告
└── references/
    ├── data-sources.md         # 数据源字段映射
    ├── classification-rules.md # 分类判定详解
    └── prompt-patterns.md      # 提示词优化模板
```

## 最佳实践

1. **定期运行**：建议每天自动跑一次全量分析，跟踪失败率变化
2. **优先修复高频问题**：先处理影响算子数最多的分类
3. **累积学习**：将每次的 `classification.json` 归档，长期看趋势
4. **人工 review LLM_PROMPT_GAP**：该分类需要人工确认 prompt 修改方向
5. **避免自动修改代码**：本 skill 只分析不修改，所有修复需人工确认

## 扩展开发

### 添加新的失败模式

1. 在 `classify_failure.py` 的 `_GENERATOR_CODE_PATTERNS` / `_EXEC_ENV_PATTERNS` 中添加新 pattern
2. 在 `analyze_constraint.py` 中添加新的 `_check_*` 函数
3. 在 `analyze_prompt.py` 的 `_FAILURE_TO_PROMPTS` 中添加新的 prompt 改进模板

### 自定义报告

修改 `generate_report.py` 中的 `_gen_*_section` 函数即可定制报告章节。

## 故障排查

### Q: 报告提示 "DB file not found"

A: 检查 `--project-root` 路径是否正确指向 operator-agent 根目录（该目录下应有 `data/operator_agent.db`）。

### Q: 用例产物都识别不到

A: 检查 `batch_cases/` 和 `cases/` 目录的文件名是否符合 `{operator}_cases.json` 模式。case_subgraph 节点产物有 `_cases.json` 后缀但文件名中含产品名，仍可被识别。

### Q: 分类结果全部是 UNKNOWN

A: 说明失败原因不在已有 pattern 中，需要：
1. 查看完整日志，确认异常类型
2. 在 `classify_failure.py` 的 `_GENERATOR_CODE_PATTERNS` 中添加新 pattern
3. 或者提供更详细的日志（可临时提高 log level）

### Q: HTML 报告样式错乱

A: HTML 生成器使用内置 markdown→html 简单转换，复杂嵌套表格可能渲染不好。可改用 `--format md` 配合外部 markdown 渲染工具（如 grip）。