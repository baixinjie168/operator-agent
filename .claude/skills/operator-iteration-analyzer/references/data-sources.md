# 数据源说明

本 skill 依赖 operator-agent 项目的多个数据源。本文档详细说明每个数据源的字段、读取方式、兼容性。

## 1. SQLite 数据库

**路径**：`data/operator_agent.db`

**核心表**：

### `document_versions`

存储算子文档的解析结果和最终约束。

```sql
CREATE TABLE document_versions (
    id              INTEGER PRIMARY KEY,
    operator_id     INTEGER REFERENCES operators(id),
    version         INTEGER DEFAULT 1,
    content         TEXT NOT NULL,           -- 原始 Markdown 内容
    content_hash    TEXT NOT NULL,
    parsed_data     TEXT,
    product_support TEXT,
    function_explanation_summary TEXT,
    json_constraints TEXT,                    -- ★ 关键字段：完整约束 JSON
    created_at      TEXT
);
```

**json_constraints 字段结构**（由 assemble_result 节点写入）：

```json
{
    "operator_name": "aclnnAbs",
    "function_explanation": "...",
    "product_support": ["Atlas_A2", "Atlas_A3"],
    "function_signature": "aclnnAbsGetWorkspaceSize(...)",
    "deterministic_computing": {...},
    "inputs": {
        "x": {
            "type": {"value": "aclTensor", "src_text": ""},
            "format": {"value": ["ND"], "src_text": ""},
            "dimensions": {"value": [0, 8], "src_text": "0-8"},
            "dtype": {"value": ["FLOAT16", "FLOAT32"], "src_text": ""},
            "is_optional": {"value": false},
            ...
        }
    },
    "outputs": {
        "y": {...}
    },
    "constraints_in_parameters": {
        "Atlas_A2": [
            {"expr_type": "type_equality", "expr": "y.dtype == x.dtype",
             "relation_params": ["x", "y"], "src_text": "y 与 x 同类型"}
        ]
    }
}
```

### `pipeline_runs`

记录每次 pipeline 执行的元信息（用于日志缺失时 fallback）。

```sql
CREATE TABLE pipeline_runs (
    id, run_id, operator_name, status, error,
    task_type, task_name, parent_task_id,
    result_json, content_hash, created_at, completed_at, doc_id
);
```

**关键字段**：
- `task_type`: `constraint_extract` / `case_generate` / `test_execute`
- `status`: `running` / `completed` / `failed`
- `error`: 失败时的错误描述

### `parameters`

存储每个参数的细粒度提取结果（用于生成代码侧 debug）。

```sql
SELECT param_name, param_type, dtype_desc, shape, src_content, ...
FROM parameters WHERE doc_id = ?;
```

## 2. 用例产物

**路径模式**：

| 节点 | 路径模板 | 文件名示例 |
|------|---------|-----------|
| generate_cases（老） | `batch_cases/{operator}_cases.json` 或 `cases/{operator}_cases.json` | `batch_cases/aclnnAbs_cases.json` |
| case_subgraph（新） | `cases/{operator}_{product_safe}_cases.json` | `cases/aclnnAbs_Atlas_A2_训练系列产品_Atlas_A2_推理系列产品_cases.json` |

**`{product_safe}` 的生成规则**（来自 `case_subgraph/generate.py`）：

```python
def _sanitize_product_name(product: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', '_', product)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe or "default"
```

注意：与 `batch_results/`（用 `_` 分隔）路径不同，scan_operators.py 会同时扫描这几种路径。

## 3. 执行结果

**路径**：`execution_results/{operator}/result.json`

**result.json 结构**：

```json
{
    "operator_name": "aclnnAbs",
    "task_report_data": {
        "report_records": [
            {
                "id": 0,
                "run_result": "SUCCESS",
                "failure_reason": null
            },
            {
                "id": 1,
                "run_result": "FAILED",
                "failure_reason": "shape mismatch: expected (1,3) got (1,4)"
            }
        ]
    }
}
```

**兼容性**：也支持顶层 `report_records` 字段（不带 `task_report_data` 包装）。

**run_result 枚举**：
- `SUCCESS` — 执行通过
- `FAILED` — 执行失败，failure_reason 含详细原因
- `SKIPPED` / `UNKNOWN` — 其他状态

## 4. 生成日志

**路径**：`logs/generate_case_{operator_name}.log`

**关键内容**：
- Python traceback（用于定位 GENERATOR_CODE_BUG）
- LLM 调用记录（输入 prompt + 输出 + token 用量）
- 阶段耗时统计

**Fallback**：如果日志文件不存在，使用 `pipeline_runs.error` 字段。

## 5. 算子源文档

**路径**：`operators/{operator_name}.md`

**Markdown 结构**：
- `# 算子名称` — 标题
- `## 功能说明` — 包含计算公式（LaTeX）
- `## 参数说明` — 包含参数表 + 使用说明
- `## 返回值说明`
- `## 约束说明`
- `## 调用示例`

**参数表表头**（识别关键）：

```
| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
```

## 6. 提示词和节点代码

| 类型 | 路径 |
|------|------|
| 提示词常量 | `packages/agent/src/agent/prompts/system.py` |
| LangGraph 节点 | `packages/agent/src/agent/nodes/*.py` |
| 生成器 | `packages/agent/src/agent/generators/*.py` |
| 状态定义 | `packages/agent/src/agent/nodes/state.py` |
| LLM 客户端 | `packages/agent/src/agent/mcp_client.py` |

## 读取优先级

skill 读取数据时遵循以下优先级（高优先级找不到时降级到低优先级）：

1. **用例产物** → 直接判断成功/失败
2. **execution_results/result.json** → 直接读 report_records
3. **logs/generate_case_*.log** → 提取 traceback 和错误
4. **document_versions.json_constraints** → 约束内容
5. **document_versions.content** → 算子源文档（用于校对约束）
6. **pipeline_runs** → 任务状态（log 缺失时 fallback）
7. **parameters / param_relations 表** → 字段级 debug