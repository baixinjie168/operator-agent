# 算子测试智能体平台 — 架构改造设计方案

> **版本**: v3.0  
> **日期**: 2026-06-09  
> **状态**: 设计确认，准备实施  

---

## 目录

1. [设计目标](#1-设计目标)
2. [任务模型设计](#2-任务模型设计)
3. [测试用例与执行结果存储](#3-测试用例与执行结果存储)
4. [LLM 意图识别系统](#4-llm-意图识别系统)
5. [数据库变更汇总](#5-数据库变更汇总)
6. [后端接口变更](#6-后端接口变更)
7. [前端交互设计](#7-前端交互设计)
8. [实施计划](#8-实施计划)
9. [附录：现有架构参考](#9-附录现有架构参考)

---

## 1. 设计目标

### 1.1 当前问题

| 问题 | 现状 |
|------|------|
| 任务概念模糊 | `pipeline_runs` 无 `task_type`，无法区分"约束提取"、"用例生成"、"测试执行" |
| 数据关联断裂 | 测试用例以整个 JSON 数组存储，与任务无关联；执行结果仅存于 `result_json` |
| 任务列表不可读 | 无法看出一个任务做了什么，产出是什么 |
| 用户操作无引导 | 无前置条件检查，无智能引导，用户需手动理解流程 |
| 数据流不统一 | 完整流水线和独立子任务两种模式并存，run 之间无关联 |

### 1.2 改造目标

1. **建立清晰的任务概念** — 每次用户操作 = 一个任务，任务间通过依赖链关联
2. **用例与执行结果结构化存储** — 单条用例一条记录，与任务和约束建立关联
3. **LLM 驱动的智能交互** — 用户自然语言输入，系统识别意图、检查条件、引导操作
4. **完整的任务可追溯性** — 支持查看任意任务的完整依赖链

### 1.3 设计原则

- **不兼容旧数据** — 可清空旧 DB，不做数据迁移
- **扩展 pipeline_runs 表** — 新增 task_type/task_name/parent_task_id 三个字段，复用现有基础设施
- **按需创建任务** — 执行到哪个步骤才创建对应任务，不预创建
- **并发安全** — 同一算子可同时跑多个任务，用 run_id 隔离
- **最小改动** — 不拆分独立子图，在 `_run_pipeline` 内按阶段判断创建任务

---

## 2. 任务模型设计

### 2.1 核心设计决策

| 决策点 | 选定方案 | 说明 |
|--------|---------|------|
| 任务表 | **扩展 `pipeline_runs` 表** | 新增 task_type/task_name/parent_task_id，复用现有 SSE/事件基础设施 |
| 任务粒度 | **每次操作 = 一个任务 + 关联链** | 兼顾操作边界清晰和依赖可追溯 |
| full_pipeline | **自动拆成多个独立任务，按需创建** | 在 `_run_pipeline` 内按阶段判断，不拆分独立子图 |
| 任务名称 | **系统自动生成** | 格式：`{算子名} {操作类型} #{序号}` |
| 依赖链 | **支持完整依赖链查看** | 任务3 → 任务2 → 任务1 |
| parent 查找 | **取最新已完成的同类型任务** | 如生成用例时取最新的 completed constraint_extract 任务 |
| 并发 | **支持同一算子并发任务** | run_id 隔离，互不影响 |
| 删除 | **支持删除任务，级联删除** | 删除任务时级联删除关联的用例和执行结果 |
| 失败处理 | **前置任务失败则不创建后续任务** | case_generate 失败 → 不创建 test_execute，提示用户 |

### 2.2 任务类型定义

| task_type | 中文名 | 触发场景 | 前置条件 | 产出 |
|-----------|--------|---------|---------|------|
| `constraint_extract` | 约束提取 | 上传文档 / 重新提取约束 | 有算子文档 | 约束 JSON + 参数列表 |
| `case_generate` | 用例生成 | 生成测试用例 | 有约束数据 | N 条测试用例 |
| `test_execute` | 测试执行 | 执行测试 | 有测试用例 | 执行结果 |

### 2.3 任务动态创建流程

**关键原则：任务按需创建，不预创建。在 `_run_pipeline` 内按阶段判断。**

**不拆分独立子图**，保持当前 LangGraph 图结构不变，在 `_run_pipeline` 函数内按阶段插入任务创建逻辑：

```python
async def _run_pipeline(run_id, operator_name, content, ...):
    # 阶段1: 约束提取（创建 pipeline_runs 记录，task_type=constraint_extract）
    constraint_task_id = create_task(operator_name, "constraint_extract", parent=None)
    # ... 执行 doc + constraint 节点 ...
    complete_task(constraint_task_id, status="completed")

    # 阶段2: 用例生成（自动创建新任务）
    case_task_id = create_task(operator_name, "case_generate", parent=constraint_task_id)
    # ... 执行 case 节点 ...
    if case_failed:
        fail_task(case_task_id, error="...")
        return  # 不创建 test_execute
    complete_task(case_task_id, status="completed")

    # 阶段3: 测试执行（自动创建新任务）
    exec_task_id = create_task(operator_name, "test_execute", parent=case_task_id)
    # ... 执行 exec 节点 ...
```

**上传文档触发全流程时的行为：**

```
上传文档 → _run_pipeline 创建 constraint_extract 任务并开始执行
    │
    ├── constraint_extract 完成 → 自动创建 case_generate 任务
    │   │
    │   ├── case_generate 完成 → 自动创建 test_execute 任务
    │   │   │
    │   │   ├── test_execute 完成 → 全流程结束
    │   │   └── test_execute 失败 → 提示失败原因
    │   │
    │   └── case_generate 失败 → 停止，不创建 test_execute
    │
    └── constraint_extract 失败 → 停止，不创建后续任务
```

**用户单独触发操作时的行为：**

```
用户点击"生成用例" → 检查约束是否存在
    ├── 存在 → 创建 case_generate 任务 (parent: 该算子最新已完成的 constraint_extract 任务)
    └── 不存在 → 引导用户先提取约束

用户点击"执行测试" → 检查用例是否存在
    ├── 存在 → 创建 test_execute 任务 (parent: 该算子最新已完成的 case_generate 任务)
    └── 不存在 → 引导用户先生成用例
```

**parent_task_id 查找规则：**

```python
def find_parent_task(operator_name: str, parent_type: str) -> str | None:
    """查找该算子最新已完成的指定类型任务"""
    row = db.execute(
        "SELECT run_id FROM pipeline_runs "
        "WHERE operator_name = ? AND task_type = ? AND status = 'completed' "
        "ORDER BY created_at DESC LIMIT 1",
        (operator_name, parent_type)
    ).fetchone()
    return row[0] if row else None
```

### 2.4 任务名称自动生成规则

```python
def generate_task_name(operator_name: str, task_type: str) -> str:
    """生成任务名称，如 'aclnnAdaLayerNorm 约束提取 #3'"""
    type_labels = {
        "constraint_extract": "约束提取",
        "case_generate": "用例生成",
        "test_execute": "测试执行",
    }
    label = type_labels.get(task_type, task_type)
    seq = count_same_type_tasks(operator_name, task_type) + 1
    return f"{operator_name} {label} #{seq}"
```

### 2.5 依赖链查询

```python
def get_task_chain(run_id: str) -> list[dict]:
    """获取任务的完整依赖链: task → parent → grandparent → ..."""
    chain = []
    current_id = run_id
    while current_id:
        task = query_run(current_id)
        if not task:
            break
        chain.append(task)
        current_id = task.get("parent_task_id")
    return chain  # [当前任务, 父任务, 祖父任务, ...]
```

### 2.6 pipeline_runs 表扩展

在现有 `pipeline_runs` 表基础上新增 3 个字段：

```sql
-- 新增字段
ALTER TABLE pipeline_runs ADD COLUMN task_type TEXT;           -- constraint_extract | case_generate | test_execute
ALTER TABLE pipeline_runs ADD COLUMN task_name TEXT;           -- 系统自动生成
ALTER TABLE pipeline_runs ADD COLUMN parent_task_id TEXT REFERENCES pipeline_runs(run_id);

-- 新增索引
CREATE INDEX IF NOT EXISTS idx_runs_task_type ON pipeline_runs(task_type);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON pipeline_runs(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_runs_operator_status ON pipeline_runs(operator_name, status);
```

**完整表结构（改造后）：**

```sql
CREATE TABLE pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL UNIQUE,           -- UUID hex[:12]，即任务 ID
    operator_id     INTEGER REFERENCES operators(id),
    doc_id          INTEGER REFERENCES document_versions(id),
    operator_name   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running', -- running | completed | failed
    content_hash    TEXT NOT NULL,
    result_json     TEXT,                           -- 各阶段产出数据
    error           TEXT,
    -- 新增字段 --
    task_type       TEXT,                           -- constraint_extract | case_generate | test_execute
    task_name       TEXT,                           -- 系统自动生成
    parent_task_id  TEXT REFERENCES pipeline_runs(run_id),
    -- 时间戳 --
    created_at      TEXT DEFAULT (datetime('now')),
    completed_at    TEXT
);
```

### 2.7 为什么扩展 pipeline_runs 而不是新建 tasks 表

| 维度 | 新建 tasks 表 | 扩展 pipeline_runs ✅ |
|------|-------------|-------------------|
| 数据一致性 | 两表同 UUID，需双写同步 | 单表单行，天然一致 |
| 代码复杂度 | 每次操作写两张表 | 只写一张表 |
| SSE 事件流 | 需额外关联 | 已有完整基础设施（RuntimeManager, EventBus, SSE） |
| 查询 | 任务列表查 tasks，事件查 pipeline_runs | 一张表搞定 |
| 删除 | 两张表都要删 | 只删一张表 |
| 现有代码改动 | 所有路由加 tasks 写入 | 路由加几个字段 |

**核心结论：一个任务 = 一次流水线运行，是同一个实体，不应拆两张表。**

### 2.8 任务删除（级联）

```python
def delete_task(run_id: str) -> dict:
    """删除任务及其所有关联数据"""
    # 1. 查找所有子任务（递归）
    child_tasks = find_all_descendant_tasks(run_id)
    all_run_ids = [run_id] + [t["run_id"] for t in child_tasks]

    # 2. 级联删除（按依赖顺序，子任务先删）
    for rid in reversed(all_run_ids):
        db.execute("DELETE FROM exec_results WHERE task_id = ?", (rid,))
        db.execute("DELETE FROM test_cases WHERE task_id = ?", (rid,))
        db.execute("DELETE FROM pipeline_events WHERE run_id = ?", (rid,))
        db.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (rid,))

    return {"deleted_tasks": len(all_run_ids)}
```

---

## 3. 测试用例与执行结果存储

### 3.1 核心设计决策

| 决策点 | 选定方案 | 说明 |
|--------|---------|------|
| 用例字段 | **JSON 对象存储完整用例** | `case_data` JSON 字段存储完整内容，不拆分 |
| 历史保留 | **保留历史用例，task_id 区分** | 重新生成不覆盖，通过 task_id 区分不同批次 |
| 展示 | **仅展示当前任务关联的用例** | 默认展示最新一次任务的用例 |
| 执行结果 | **独立 exec_results 表** | 同一用例可执行多次，每次一条记录 |
| 约束关联 | **记录 constraint_doc_id** | 用例基于哪版约束生成，跟算子版本 ID 关联 |
| 本地文件 | **删除本地文件存储逻辑** | 执行时从数据库查询用例数据生成传输 |
| CPU精度 | **TRUE/FALSE/NULL** | 不用 0/1，TRUE=通过，FALSE=未通过，NULL=不涉及。当前阶段模拟全部 TRUE |

### 3.2 test_cases 表

```sql
CREATE TABLE test_cases (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id           TEXT NOT NULL REFERENCES pipeline_runs(run_id),  -- 生成该用例的任务
    operator_name     TEXT NOT NULL,
    case_index        INTEGER NOT NULL,          -- 批次内序号 (0, 1, 2, ...)
    case_name         TEXT NOT NULL,
    case_data         TEXT NOT NULL,             -- 完整用例 JSON 对象
    constraint_doc_id INTEGER REFERENCES document_versions(id),
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_test_cases_task ON test_cases(task_id);
CREATE INDEX IF NOT EXISTS idx_test_cases_operator ON test_cases(operator_name);
CREATE INDEX IF NOT EXISTS idx_test_cases_constraint_doc ON test_cases(constraint_doc_id);
```

**case_data JSON 结构示例**：

```json
{
    "name": "aclnnAdaLayerNorm",
    "api": "aclnnAdaLayerNorm",
    "api_type": "normal",
    "version": "9.0.0",
    "backward": false,
    "inputs": [
        {"name": "x", "dtype": "FLOAT16", "shape": [2, 3, 4], "format": "ND"},
        {"name": "gamma", "dtype": "FLOAT16", "shape": [4], "format": "ND"},
        {"name": "beta", "dtype": "FLOAT16", "shape": [4], "format": "ND"}
    ],
    "outputs": [
        {"name": "y", "dtype": "FLOAT16", "shape": [2, 3, 4], "format": "ND"}
    ],
    "attrs": {
        "epsilon": 1e-5,
        "normalized_shape": [4]
    }
}
```

### 3.3 exec_results 表

```sql
CREATE TABLE exec_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL REFERENCES pipeline_runs(run_id),  -- 执行该结果的任务
    case_id             INTEGER NOT NULL REFERENCES test_cases(id),      -- 关联用例
    operator_name       TEXT NOT NULL,
    passed              INTEGER NOT NULL,          -- 0=fail, 1=pass
    cpu_precision_passed INTEGER,                  -- TRUE=通过, FALSE=未通过, NULL=不涉及 (当前模拟全部 TRUE)
    precision_detail    TEXT,                      -- 精度详情 (误差值、阈值等)
    actual_json         TEXT,                      -- 实际输出 JSON
    error_message       TEXT,                      -- 错误信息
    cpu_reference_code  TEXT,                      -- CPU 参考实现代码
    duration_ms         INTEGER,                   -- 执行耗时(毫秒)
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_exec_results_task ON exec_results(task_id);
CREATE INDEX IF NOT EXISTS idx_exec_results_case ON exec_results(case_id);
CREATE INDEX IF NOT EXISTS idx_exec_results_operator ON exec_results(operator_name);
```

### 3.4 数据关系全景

```
operators (算子)
    │
    ├── document_versions (文档版本, 1:N)
    │   ├── content (原始 Markdown)
    │   ├── parsed_data (结构化解析)
    │   ├── json_constraints (约束 JSON)
    │   │
    │   ├── parameters (参数, 1:N)
    │   ├── param_relations (参数关系, 1:N)
    │   ├── function_signatures (函数签名, 1:N)
    │   ├── platform_support (平台支持, 1:N)
    │   ├── return_codes (返回码, 1:N)
    │   └── dtype_combinations (数据类型组合, 1:N)
    │
    └── pipeline_runs (任务, 1:N)
        │
        ├── task_type = "constraint_extract"
        │   └── doc_id → document_versions
        │
        ├── task_type = "case_generate"
        │   ├── parent_task_id → 约束提取任务 (run_id)
        │   └── test_cases (用例, 1:N)
        │       └── constraint_doc_id → 约束所属文档版本
        │
        └── task_type = "test_execute"
            ├── parent_task_id → 用例生成任务 (run_id)
            └── exec_results (执行结果, 1:N)
                └── case_id → test_cases
```

### 3.5 典型查询场景

| 场景 | SQL |
|------|-----|
| 查看某任务生成的用例 | `SELECT * FROM test_cases WHERE task_id = ? ORDER BY case_index` |
| 查看算子最新一批用例 | `SELECT * FROM test_cases WHERE task_id = (SELECT run_id FROM pipeline_runs WHERE operator_name = ? AND task_type = 'case_generate' AND status = 'completed' ORDER BY created_at DESC LIMIT 1) ORDER BY case_index` |
| 查看某次执行的详细结果 | `SELECT er.*, tc.case_name, tc.case_data FROM exec_results er JOIN test_cases tc ON er.case_id = tc.id WHERE er.task_id = ?` |
| 查看某条用例的历史执行 | `SELECT * FROM exec_results WHERE case_id = ? ORDER BY created_at DESC` |
| 执行时获取用例数据 | `SELECT case_data FROM test_cases WHERE task_id = ? ORDER BY case_index` |

---

## 4. LLM 意图识别系统

### 4.1 核心设计决策

| 决策点 | 选定方案 | 说明 |
|--------|---------|------|
| 解析位置 | **后端** | 统一管控，前端只需传文本 |
| 解析策略 | **规则优先 + LLM 兜底** | 常见模式用正则快速匹配，匹配不到再调 LLM |
| 确认交互 | **仅写操作确认** | 查看类操作直接执行，生成/执行类操作需确认 |
| 复合指令 | **拆成多个独立任务串行执行** | "生成用例并执行" → case_generate + test_execute |
| 多轮对话 | **支持（内存会话）** | 刷新页面丢失，每次都是新会话 |
| 约束版本 | **自动用最新** | 后续考虑多版本再做调整 |
| 意图错误 | **回复"没理解" + 建议** | 让用户重新尝试 |

### 4.2 整体架构

```
用户输入 (自然语言)
    │
    ▼
┌──────────────────────────────────────────┐
│  POST /api/v1/chat/parse-intent          │
│                                          │
│  1. 规则引擎快速匹配                    │
│     正则匹配常见模式 → 命中则直接返回     │
│                                          │
│  2. LLM 意图解析（规则未命中时）          │
│     输入: 用户文本 + 上下文               │
│     输出: 结构化意图 JSON                 │
│                                          │
│  3. 前置条件检查                          │
│     查DB: 算子存在? 约束存在? 用例存在?    │
│                                          │
│  4. 生成响应策略                          │
│     条件满足 → response_type: "confirm"   │
│     条件不满足 → response_type: "guide"   │
│     算子不存在 → response_type: "error"   │
│     无法识别 → response_type: "unknown"   │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  前端渲染交互卡片                         │
│                                          │
│  confirm → 确认卡片 [取消] [确认执行]     │
│  guide   → 引导卡片 [自动处理并继续]      │
│  error   → 错误卡片 [上传文档] [选择算子]  │
│  direct  → 直接执行（只读操作）           │
│  unknown → "没理解" + 建议操作列表        │
└──────────────────────────────────────────┘
```

### 4.3 意图定义

```python
class UserIntent(BaseModel):
    action: str                # 动作类型
    operator_name: str | None  # 识别到的算子名（可能为空）
    confidence: float          # 置信度 0-1
    parameters: dict           # 附加参数

# 支持的 action 类型
ACTIONS = {
    # 只读操作（直接执行，无需确认）
    "view_constraints":    "查看算子约束",
    "view_cases":          "查看测试用例",
    "view_results":        "查看执行结果",
    "view_document":       "查看算子文档",
    "list_operators":      "列出所有算子",
    "view_task_history":   "查看任务历史",
    "view_task_chain":     "查看任务依赖链",
    "help":                "帮助/使用说明",

    # 写操作（需要确认）
    "extract_constraints": "提取/重新提取约束",
    "generate_cases":      "生成测试用例",
    "execute_tests":       "执行测试",
    "generate_and_execute": "生成用例并执行（拆为两个任务）",
    "upload_document":     "上传算子文档",

    # 系统
    "unknown":             "无法识别",
}
```

### 4.4 规则引擎（优先匹配）

```python
import re

INTENT_PATTERNS = [
    # 只读操作
    (r"(?:查看|看看|查一下|打开)\s*(\w+)\s*(?:的)?约束", "view_constraints"),
    (r"(?:查看|看看|有哪些)\s*(\w+)\s*(?:的)?用例", "view_cases"),
    (r"(?:查看|看看)\s*(\w+)\s*(?:的)?(?:执行)?结果", "view_results"),
    (r"(?:查看|看看)\s*(\w+)\s*(?:的)?文档", "view_document"),
    (r"(?:有哪些|列出|所有)\s*算子", "list_operators"),
    (r"(?:历史|记录|任务列表)", "view_task_history"),
    (r"(?:帮助|怎么用|能做什么)", "help"),

    # 写操作
    (r"(?:提取|重新提取|跑一下)\s*(\w+)\s*(?:的)?约束", "extract_constraints"),
    (r"(?:生成|创建)\s*(\w+)\s*(?:的)?(?:测试)?用例\s*(?:并|然后|再)\s*(?:执行|跑)", "generate_and_execute"),
    (r"(?:生成|创建)\s*(\w+)\s*(?:的)?(?:测试)?用例", "generate_cases"),
    (r"(?:执行|运行|跑)\s*(\w+)\s*(?:的)?(?:测试)?(?:用例|测试)", "execute_tests"),
    (r"(?:上传|导入)\s*(?:算子)?文档", "upload_document"),

    # 多轮对话确认词
    (r"^(?:好的|确认|继续|执行吧|没问题|可以|ok|yes)$", "__confirm__"),
    (r"^(?:算了|取消|不要了|不了|no)$", "__cancel__"),
]

def rule_based_parse(text: str, current_operator: str | None) -> UserIntent | None:
    """规则引擎快速匹配，返回 None 表示未命中"""
    text = text.strip()
    for pattern, action in INTENT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # 提取算子名（如果有捕获组）
            op_name = None
            if match.groups():
                op_name = match.group(1)
            if not op_name:
                op_name = current_operator
            return UserIntent(
                action=action,
                operator_name=op_name,
                confidence=0.95,
                parameters={},
            )
    return None  # 未命中，交给 LLM
```

### 4.5 LLM Prompt 设计（规则未命中时使用）

```python
INTENT_PARSE_PROMPT = """你是一个昇腾算子测试平台的意图解析器。
根据用户输入和当前上下文，输出结构化意图 JSON。

## 当前上下文
- 当前选中算子: {current_operator}（可能为空）
- 已有算子列表: {operator_list}
- 对话历史: {conversation_history}

## 支持的意图

### 只读操作（直接执行）
- view_constraints: 查看约束
- view_cases: 查看用例
- view_results: 查看执行结果
- view_document: 查看文档
- list_operators: 列出算子
- view_task_history: 任务历史
- help: 帮助

### 写操作（需要确认）
- extract_constraints: 提取约束
- generate_cases: 生成用例
- execute_tests: 执行测试
- generate_and_execute: 生成用例并执行
- upload_document: 上传文档

## 规则
1. 如果用户提到了算子名称，提取出来（支持模糊匹配，如"Ada"匹配"aclnnAdaLayerNorm"）
2. 如果用户没有提到算子名称，使用当前选中算子
3. 如果都没有，operator_name 为 null
4. 多轮对话中，用户说"好的"、"继续"、"执行吧"等，结合上文确定意图
5. 复合指令如"生成并执行"识别为 generate_and_execute
6. confidence 低于 0.6 时标记为 unknown

## 输出格式（严格 JSON）
{"action": "...", "operator_name": "...", "confidence": 0.95, "parameters": {}}
"""
```

### 4.6 前置条件检查接口

```python
# GET /api/v1/operators/{name}/readiness
@router.get("/operators/{operator_name}/readiness")
async def check_operator_readiness(operator_name: str):
    """检查算子各阶段数据就绪状态"""
    return {
        "exists": True,
        "has_document": True,
        "has_constraints": True,
        "constraint_doc_id": 5,
        "constraint_version": 2,
        "parameters_count": 8,
        "has_cases": True,
        "cases_count": 20,
        "latest_cases_task_id": "abc123",
        "has_exec_results": True,
        "latest_exec_task_id": "def456",
        "latest_exec_passed": 18,
        "latest_exec_total": 20,
    }
```

### 4.7 场景矩阵

| 用户输入 | 解析结果 | 前置检查 | 系统响应 |
|---------|---------|---------|---------|
| "查看 Ada 约束" | `view_constraints` | ✅ 约束存在 | **直接执行**：弹框展示约束 |
| | | ❌ 约束不存在 | **引导**："该算子暂无约束，是否提取？[提取约束]" |
| "生成 Ada 的用例" | `generate_cases` | ✅ 约束存在 | **确认**："将为 aclnnAdaLayerNorm 生成用例，基于约束 v2，确认？[确认]" |
| | | ❌ 约束不存在 | **引导**："需要先提取约束，是否自动执行？[自动提取并继续]" |
| | | ❌ 算子不存在 | **错误**："未找到算子，请确认名称或上传文档 [上传文档]" |
| "执行 xxx 测试" | `execute_tests` | ✅ 用例存在 | **确认**："将执行 xxx 的 20 条用例，确认？[确认执行]" |
| | | ❌ 用例不存在 | **引导**："暂无用例，是否先生成？[生成用例]" |
| "xxx 生成用例并执行" | `generate_and_execute` | ✅ 约束存在 | **确认**："将为 xxx 生成用例并执行测试，确认？[确认]" |
| | | ❌ 约束不存在 | **引导**："需要完整流程：提取约束→生成用例→执行，是否继续？[全流程执行]" |
| "好的" (上文: 询问是否生成用例) | 继承上文意图 | — | 按上文意图继续执行 |
| "有哪些算子" | `list_operators` | 无 | **直接执行**：展示算子列表 |
| "帮我优化一下代码" | `unknown` | 无 | **"没理解"**：抱歉，我没理解您的意思。您可以尝试：查看约束、生成用例、执行测试等 |

### 4.8 多轮对话设计

#### 会话管理

```python
class ConversationSession:
    """内存会话，刷新页面丢失"""
    session_id: str                    # 会话 ID（页面加载时生成）
    messages: list[ChatMessage]        # 对话历史（最近 10 轮）
    current_operator: str | None       # 当前算子
    last_intent: UserIntent | None     # 上一轮意图
    pending_action: str | None         # 待确认的操作
```

- **存储位置**：后端内存（dict），key 为 session_id
- **生命周期**：页面刷新则丢失，每次打开页面是新会话
- **历史保留**：最近 10 轮对话
- **后续演进**：可考虑持久化到数据库

#### 上下文继承规则

| 用户输入模式 | 继承策略 |
|-------------|---------|
| "好的"、"确认"、"继续"、"执行吧" | 继承 `last_intent`，执行待确认操作 |
| "算了"、"取消"、"不要了" | 清除 `pending_action` |
| 新的算子名 | 更新 `current_operator` |
| 新的操作指令 | 覆盖 `last_intent` |
| 省略算子名的操作 | 使用 `current_operator` |

### 4.9 后端接口

```python
# 意图解析 + 前置条件检查（合并接口）
POST /api/v1/chat/parse-intent
Request: {
    "text": "执行 Ada 的测试用例",
    "session_id": "sess_abc123",
    "current_operator": "aclnnAdaLayerNorm"
}
Response: {
    "intent": {
        "action": "execute_tests",
        "operator_name": "aclnnAdaLayerNorm",
        "confidence": 0.92,
        "parameters": {}
    },
    "readiness": {
        "exists": true,
        "has_constraints": true,
        "has_cases": true,
        "cases_count": 20
    },
    "response_type": "confirm",     // "direct" | "confirm" | "guide" | "error" | "unknown"
    "response_message": "将执行 aclnnAdaLayerNorm 的 20 条测试用例",
    "suggested_actions": [
        {"label": "确认执行", "action": "execute_tests", "params": {"operator_name": "aclnnAdaLayerNorm"}},
        {"label": "取消", "action": "cancel"}
    ]
}
```

### 4.10 前端交互卡片

#### 确认卡片（写操作，条件满足）

```
┌──────────────────────────────────────────────┐
│ 操作确认                                      │
│                                              │
│ 将为 aclnnAdaLayerNorm 生成测试用例            │
│ 基于约束版本: v2 (8个参数, 15条约束规则)       │
│                                              │
│ [取消]                    [确认生成]          │
└──────────────────────────────────────────────┘
```

#### 引导卡片（前置条件不满足）

```
┌──────────────────────────────────────────────┐
│ 前置条件未满足                                 │
│                                              │
│ 算子 aclnnFoo 尚未提取约束                    │
│ 需要执行以下步骤:                              │
│  1. 提取算子约束                              │
│  2. 生成测试用例                              │
│                                              │
│ [取消]              [自动提取约束并继续]       │
└──────────────────────────────────────────────┘
```

#### 错误卡片（算子不存在）

```
┌──────────────────────────────────────────────┐
│ 算子未找到                                    │
│                                              │
│ 未找到名为 "aclnnFoo" 的算子                   │
│ 相似算子: aclnnFloor, aclnnFold              │
│                                              │
│ [选择相似算子]           [上传新算子文档]      │
└──────────────────────────────────────────────┘
```

#### 未知意图卡片

```
┌──────────────────────────────────────────────┐
│ 抱歉，我没理解您的意思                        │
│                                              │
│ 您可以尝试以下操作:                            │
│  - 查看 {算子名} 的约束                       │
│  - 生成 {算子名} 的测试用例                   │
│  - 执行 {算子名} 的测试                       │
│  - 查看任务历史                               │
│  - 输入"帮助"查看完整操作列表                 │
└──────────────────────────────────────────────┘
```

---

## 5. 数据库变更汇总

### 5.1 pipeline_runs 表（扩展）

```sql
-- 新增 3 个字段
ALTER TABLE pipeline_runs ADD COLUMN task_type TEXT;
ALTER TABLE pipeline_runs ADD COLUMN task_name TEXT;
ALTER TABLE pipeline_runs ADD COLUMN parent_task_id TEXT REFERENCES pipeline_runs(run_id);

-- 新增索引
CREATE INDEX IF NOT EXISTS idx_runs_task_type ON pipeline_runs(task_type);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON pipeline_runs(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_runs_operator_status ON pipeline_runs(operator_name, status);
```

### 5.2 test_cases 表（新建，替代旧表）

```sql
CREATE TABLE test_cases (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id           TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    operator_name     TEXT NOT NULL,
    case_index        INTEGER NOT NULL,
    case_name         TEXT NOT NULL,
    case_data         TEXT NOT NULL,             -- 完整用例 JSON 对象
    constraint_doc_id INTEGER REFERENCES document_versions(id),
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_test_cases_task ON test_cases(task_id);
CREATE INDEX IF NOT EXISTS idx_test_cases_operator ON test_cases(operator_name);
CREATE INDEX IF NOT EXISTS idx_test_cases_constraint_doc ON test_cases(constraint_doc_id);
```

### 5.3 exec_results 表（新建）

```sql
CREATE TABLE exec_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    case_id             INTEGER NOT NULL REFERENCES test_cases(id),
    operator_name       TEXT NOT NULL,
    passed              INTEGER NOT NULL,          -- 0=fail, 1=pass
    cpu_precision_passed INTEGER,                  -- TRUE/FALSE/NULL
    precision_detail    TEXT,
    actual_json         TEXT,
    error_message       TEXT,
    cpu_reference_code  TEXT,
    duration_ms         INTEGER,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_exec_results_task ON exec_results(task_id);
CREATE INDEX IF NOT EXISTS idx_exec_results_case ON exec_results(case_id);
CREATE INDEX IF NOT EXISTS idx_exec_results_operator ON exec_results(operator_name);
```

### 5.4 不变动的表

| 表名 | 说明 |
|------|------|
| `operators` | 不变 |
| `document_versions` | 不变 |
| `parameters` | 不变 |
| `param_relations` | 不变 |
| `function_signatures` | 不变 |
| `platform_support` | 不变 |
| `return_codes` | 不变 |
| `dtype_combinations` | 不变 |
| `constraints_result` | 不变 |
| `pipeline_events` | 不变，继续用于事件存储 |

### 5.5 废弃的旧表

| 旧表 | 替代 |
|------|------|
| `test_cases`（旧，JSON 数组存储） | 新 `test_cases` 表（单条记录） |

### 5.6 ER 关系图

```
operators
    │
    ├──< document_versions
    │        │
    │        ├── parameters, param_relations, function_signatures, ...
    │        │
    │        └──< pipeline_runs (task_type=constraint_extract, via doc_id)
    │
    └──< pipeline_runs
             │
             ├── constraint_extract
             │   doc_id → document_versions
             │
             ├── case_generate
             │   parent_task_id → constraint_extract 任务 (run_id)
             │   │
             │   └──< test_cases
             │        constraint_doc_id → document_versions
             │
             └── test_execute
                 parent_task_id → case_generate 任务 (run_id)
                 │
                 └──< exec_results
                      case_id → test_cases
```

---

## 6. 后端接口变更

### 6.1 新增接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat/parse-intent` | 意图解析（规则+LLM）+ 前置条件检查 |
| `GET` | `/api/v1/operators/{name}/readiness` | 算子数据就绪状态检查 |
| `GET` | `/api/v1/runs/{run_id}/chain` | 任务完整依赖链 |
| `DELETE` | `/api/v1/runs/{run_id}` | 删除任务（级联删除用例和结果） |
| `GET` | `/api/v1/test-cases` | 查询用例（支持 task_id、operator_name 过滤） |
| `GET` | `/api/v1/exec-results` | 查询执行结果（支持 task_id、case_id 过滤） |

### 6.2 修改接口

| 方法 | 路径 | 变更 |
|------|------|------|
| `POST` | `/api/v1/upload` | 写入 task_type/task_name/parent_task_id，按阶段动态创建后续任务 |
| `POST` | `/api/v1/generator/run` | 写入 task_type/task_name/parent_task_id，用例写入 test_cases 表 |
| `POST` | `/api/v1/execute/run` | 写入 task_type/task_name/parent_task_id，结果写入 exec_results 表，从 DB 读取用例数据 |
| `GET` | `/api/v1/runs` | 支持 task_type、operator_name 过滤 |
| `GET` | `/api/v1/operators/{name}/latest-run` | 支持按 task_type 查询最新任务 |

### 6.3 废弃接口

| 方法 | 路径 | 替代 |
|------|------|------|
| `POST` | `/api/v1/cases/generate` | 统一走 `/api/v1/generator/run` |
| `GET` | `/api/v1/cases/{operator_name}` | 改为 `/api/v1/test-cases?operator_name=xxx` |
| `GET` | `/api/v1/cases` | 改为 `/api/v1/runs?task_type=case_generate` |

---

## 7. 前端交互设计

### 7.1 用户操作流程

```
用户进入平台
    │
    ├── 方式A: 上传算子文档
    │   └── 创建 constraint_extract 任务
    │       └── 完成后自动创建 case_generate 任务
    │           └── 完成后自动创建 test_execute 任务
    │       └── 任何阶段失败则停止，不创建后续任务
    │
    ├── 方式B: 选择已有算子 → 快捷操作
    │   ├── "查看约束" → 直接弹框展示（无需确认）
    │   ├── "生成测试用例" → 检查约束 → 确认卡片 → 创建任务执行
    │   ├── "执行测试" → 检查用例 → 确认卡片 → 创建任务执行
    │   └── "重新提取约束" → 确认卡片 → 创建任务执行
    │
    └── 方式C: 自然语言输入
        └── 规则匹配 → (未命中) → LLM 意图解析
            → 前置条件检查 → 交互卡片 → 创建任务执行
```

### 7.2 任务列表

- **入口**：会话框右上角 icon 按钮
- **展示**：点击后弹出任务列表面板
- **默认展示**：当前选中算子的任务列表
- **排序**：按创建时间倒序

```
┌─────────────────────────────────────────────────────────────────────┐
│ 任务列表 — aclnnAdaLayerNorm                              [关闭]    │
├──────────┬────────────┬────────┬──────────────┬────────┬────────────┤
│ 任务ID   │ 类型        │ 状态   │ 产出         │ 耗时   │ 时间       │
├──────────┼────────────┼────────┼──────────────┼────────┼────────────┤
│ a1b2c3d4 │ 约束提取    │ 完成   │ 8参数 15约束  │ 2m 05s │ 06-09 19:10│
│  └─ e5f6 │ 用例生成    │ 完成   │ 20条用例     │ 45s    │ 06-09 19:13│
│     └─ g7│ 测试执行    │ 失败   │ 15/20通过    │ 1m 10s │ 06-09 19:15│
│  └─ h8i9 │ 用例生成    │ 完成   │ 25条用例     │ 50s    │ 06-09 19:20│
│     └─ j0│ 测试执行    │ 完成   │ 23/25通过    │ 1m 30s │ 06-09 19:22│
├──────────┴────────────┴────────┴──────────────┴────────┴────────────┤
│  缩进表示依赖关系 (└─ = 子任务)                    [删除选中任务]    │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 依赖链查看

任务列表中通过缩进直观展示依赖关系。点击任务可展开完整依赖链：

```
任务 j0k1 (测试执行 #2)
  └── 来源: h8i9 (用例生成 #2)
       └── 来源: a1b2 (约束提取 #1)
            └── 根任务
```

### 7.4 用例展示

- **默认展示**：最新一次 case_generate 任务的用例
- **切换**：可在任务列表中选择特定任务查看其关联的用例
- **不再保存本地文件**：执行时从数据库查询用例数据

---

## 8. 实施计划

### 8.1 开发顺序（按依赖关系）

```
阶段1: 任务模型 (基础)
    ├── pipeline_runs 表新增 task_type/task_name/parent_task_id 字段
    ├── 修改 upload/generator/execute 路由写入新字段
    ├── _run_pipeline 内按阶段动态创建后续任务（不拆分子图）
    ├── 实现任务名称自动生成
    ├── 实现依赖链查询 (GET /runs/{id}/chain)
    ├── 实现任务删除（级联）(DELETE /runs/{id})
    └── 前端：任务列表 icon + 面板

阶段2: 用例存储 (依赖阶段1)
    ├── 新建 test_cases 表
    ├── 新建 exec_results 表
    ├── 修改 generator 路由：用例逐条写入 test_cases
    ├── 修改 execute 路由：从 DB 读取用例，结果写入 exec_results
    ├── 删除本地文件存储逻辑
    ├── 实现用例/执行结果查询接口
    └── 前端：用例展示（按任务关联）

阶段3: 意图识别 (依赖阶段1+2)
    ├── 实现规则引擎（正则匹配）
    ├── 实现 LLM 意图解析接口
    ├── 实现前置条件检查接口
    ├── 实现多轮对话会话管理（内存）
    ├── 实现交互卡片（确认/引导/错误/未知）
    └── 前端：会话框集成意图识别
```

### 8.2 各阶段预估工作量

| 阶段 | 后端 | 前端 | 合计 |
|------|------|------|------|
| 阶段1: 任务模型 | 3-4天 | 2-3天 | 5-7天 |
| 阶段2: 用例存储 | 2-3天 | 1-2天 | 3-5天 |
| 阶段3: 意图识别 | 3-4天 | 2-3天 | 5-7天 |
| **合计** | **8-11天** | **5-8天** | **13-19天** |

---

## 9. 附录：现有架构参考

### 改造后 pipeline_runs 完整表结构

```sql
CREATE TABLE pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL UNIQUE,           -- UUID hex[:12]，即任务 ID
    operator_id     INTEGER REFERENCES operators(id),
    doc_id          INTEGER REFERENCES document_versions(id),
    operator_name   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running', -- running | completed | failed
    content_hash    TEXT NOT NULL,
    result_json     TEXT,
    error           TEXT,
    -- 新增字段 --
    task_type       TEXT,                           -- constraint_extract | case_generate | test_execute
    task_name       TEXT,                           -- 系统自动生成
    parent_task_id  TEXT REFERENCES pipeline_runs(run_id),
    -- 时间戳 --
    created_at      TEXT DEFAULT (datetime('now')),
    completed_at    TEXT
);
```

### 三种入口（改造后）

| 入口 | 路由 | task_type | 行为 |
|------|------|-----------|------|
| 上传文档 | `POST /api/v1/upload` | `constraint_extract` | 创建约束提取任务，完成后自动串联后续阶段 |
| 生成用例 | `POST /api/v1/generator/run` | `case_generate` | 独立任务，parent 指向最新 constraint_extract |
| 执行测试 | `POST /api/v1/execute/run` | `test_execute` | 独立任务，parent 指向最新 case_generate |

### 当前 Agent 与节点映射（不变）

| agent_id | 节点 |
|----------|------|
| `doc` | init_doc, parse_params, product_support, function_*_extract, src_content_extract, param_desc_extract, shape/dtype/optional/dformat/param_attr/array_length/allowed_range/return_code/determinism/dtype_combo_extract, param_relation_extract |
| `constraint` | build_param_relations, build_param_constraint, assemble_result |
| `case` | case_match_model, case_init_static, case_solve_constraints, case_generate |
| `execute` | exec_generate_atk, exec_cpu_derivation, exec_run_atk |
