# Operator Agent 项目总览

> 生成日期: 2026-05-25 | 基于代码实际状态，非 CLAUDE.md 中的理想架构

---

## 1. 项目定位

基于 **LangGraph + MCP** 的 CANN 算子文档自动化处理系统。将华为昇腾平台的算子 Markdown 文档解析为结构化数据，经 LLM 提取参数、约束、产品支持信息，最终生成 Python 测试用例。

**当前完成度**：约 30%，文档解析、数据库层、基础 Pipeline 可运行。约束提取、人工审核、测试用例生成尚未实现。

---

## 2. 技术栈

| 层级 | 技术 | 备注 |
|------|------|------|
| 语言 | Python 3.12+ | |
| 包管理 | uv (monorepo, 3 个 hatchling 子包) | |
| Web 框架 | FastAPI + Uvicorn | |
| 工作流引擎 | LangGraph (StateGraph) | 确定性 DAG，非 LLM Agent |
| LLM 调用 | langchain-openai (ChatOpenAI) | 支持 Z.AI (GLM) 和 DeepSeek |
| 协议桥接 | MCP Python SDK (FastMCP, stdio transport) | Agent ↔ MCP Server 通信 |
| 数据验证 | Pydantic v2 + pydantic-settings | |
| 数据库 | SQLite (同步, WAL 模式) | 注意：依赖声明的 aiosqlite 未使用 |
| Markdown 解析 | markdown-it-py | 仅用于 token 级别的章节分割 |
| 前端 | 单页 HTML/JS (内嵌于 static/index.html) | 无前端构建工具 |
| 测试 | pytest + pytest-asyncio + pytest-cov | 仅 mcp_server 有测试 |
| 类型检查 | mypy (strict) | |
| Lint/Format | ruff + isort | |

---

## 3. 目录结构与模块职责

```
operator-agent/
├── packages/
│   ├── shared/                      # 领域模型层（纯数据结构，无业务逻辑）
│   │   └── src/shared/
│   │       ├── exceptions.py        # 异常层次：OperatorAgentError 及 4 个子类
│   │       └── models/
│   │           ├── enums.py         # 9 个 StrEnum（DataType, SectionType, LLMProvider 等）
│   │           ├── operator.py      # 6 个文档解析模型（ParsedOperatorDocument, ParsedSection 等）
│   │           ├── constraint.py    # 7 个约束模型（ShapeConstraint, OperatorConstraint 等）
│   │           └── test_case.py     # 3 个测试用例模型（TensorSpec, TestCase, TestFile）
│   │
│   ├── mcp-server/                  # 数据访问层（唯一直接访问 SQLite 的服务）
│   │   └── src/mcp_server/
│   │       ├── __main__.py          # 入口：初始化 DB，调用 mcp.run()
│   │       ├── db.py                # Database 类 + get_db() 单例，同步 SQLite + WAL
│   │       ├── schema.sql           # 3 张表：operators, document_versions, parameters
│   │       ├── server.py            # FastMCP 实例，注册 13 个 tool + 2 个 resource
│   │       ├── parsers/
│   │       │   ├── section_splitter.py  # 基于 markdown-it-py token 的章节分割
│   │       │   └── document_parser.py   # 章节分类 + 产品支持表解析 + 函数签名提取
│   │       └── tools/
│   │           └── document_tools.py    # 13 个 MCP tool 实现函数（纯同步）
│   │
│   └── agent/                       # 应用层（FastAPI + LangGraph Pipeline）
│       └── src/agent/
│           ├── main.py              # create_app() 工厂函数，挂载路由/静态文件
│           ├── graph.py             # create_pipeline_graph() — LangGraph 图定义
│           ├── mcp_client.py        # MCPClient — 通过 stdio 子进程调用 MCP Server
│           ├── core/
│           │   ├── config.py        # pydantic-settings，双 LLM 提供商切换
│           │   └── logging.py       # setup_logging + get_logger
│           ├── nodes/
│           │   ├── state.py         # PipelineState TypedDict（带 merge_errors reducer）
│           │   ├── init_doc.py      # 节点1：版本检查 → 保存 → 解析
│           │   ├── product_support.py   # 节点2a（并行）：LLM 提取产品支持表
│           │   ├── parse_params.py      # 节点2b（并行）：LLM 提取函数参数
│           │   └── param_desc_extract.py # 节点3：LLM 并发提取参数描述
│           ├── routes/
│           │   ├── upload.py        # POST /api/v1/upload — 触发 Pipeline
│           │   └── query.py         # GET /api/v1/operators, /operators/{name}, /parameters
│           ├── schemas/
│           │   ├── upload.py        # UploadResponse
│           │   └── query.py         # OperatorListResponse, OperatorDetailResponse, ParameterListResponse
│           ├── prompts/
│           │   ├── __init__.py      # PRODUCT_SUPPORT_EXTRACT_PROMPT, PARAM_DESC_EXTRACT_PROMPT
│           │   └── system.py        # SYSTEM_PROMPT（未使用）
│           ├── tools/
│           │   ├── document_tools.py    # @tool 装饰的 LangGraph 工具（节点未使用，直接用 MCPClient）
│           │   └── section_tools.py     # 存根工具（返回 "not yet implemented"）
│           └── static/
│               └── index.html       # 单页测试管理 UI（文档解析 + 查询 + 参数表）
│
├── operators/                       # ~510 个 CANN 算子 Markdown 源文档
│   ├── (root)/, cv/, math/, nn/, trans/
├── tests/                           # 测试（仅 mcp_server/parsers 有 25 个测试）
├── docs/                            # 设计文档 + LLM 提取提示指南
│   ├── design.md                    # 完整设计文档 v0.2
│   ├── plan.md                      # 实现计划 v1.0
│   └── prompts/                     # 7 个提取阶段的提示模板
├── data/operator_agent.db           # SQLite 数据库（运行时生成）
├── CLAUDE.md                        # 项目级 Claude Code 指引
├── pyproject.toml                   # 根工作区配置（ruff, mypy, pytest）
└── .env                             # 环境变量（LLM API Key 等，gitignored）
```

---

## 4. 入口文件

| 入口 | 路径 | 启动方式 |
|------|------|---------|
| FastAPI 应用 | `packages/agent/src/agent/main.py` | `uvicorn agent.main:create_app --factory --reload` |
| MCP Server | `packages/mcp-server/src/mcp_server/__main__.py` | `python -m mcp_server` 或 `uv run python -m mcp_server` |
| Pipeline 触发 | `packages/agent/src/agent/routes/upload.py` | 通过 `POST /api/v1/upload` 调用 |
| 前端页面 | `packages/agent/src/agent/static/index.html` | 访问 `http://localhost:8000/` |

---

## 5. 核心流程：文档上传 Pipeline

### 5.1 整体架构

```
用户上传 .md 文件
        │
        ▼
  FastAPI upload_document()
    - 提取 operator_name（正则匹配标题）
    - 计算 SHA256 content_hash
    - 构造初始状态 {operator_name, content, content_hash}
        │
        ▼
  graph.ainvoke(initial_state)
        │
        ▼
  ┌──────────────────────────────────────────────────┐
  │              LangGraph Pipeline                   │
  │                                                  │
  │  START                                           │
  │    │                                             │
  │    ▼                                             │
  │  init_doc           (顺序，第1步)                  │
  │    │                                             │
  │    ├── error ───────────► END                    │
  │    │                                             │
  │    ▼ (status = new/unchanged/updated)            │
  │  ┌─────────────────────┐                         │
  │  │   PARALLEL FAN-OUT   │                        │
  │  │   (同时执行两个节点)    │                        │
  │  ├─────────┬───────────┤                         │
  │  ▼         ▼           │                         │
  │  product   parse       │                         │
  │  _support  _params     │                         │
  │  (并行2a)   (并行2b)    │                         │
  │  │         │           │                         │
  │  └────┬────┘           │                         │
  │       │ (两者都完成后才继续) │                        │
  │       ▼                │                         │
  │  param_desc_extract    (顺序，第3步)               │
  │       │                │                         │
  │       ▼                │                         │
  │      END               │                         │
  └──────────────────────────────────────────────────┘
        │
        ▼
  返回最终 state → UploadResponse
```

### 5.2 各节点详细职责

#### 节点1: init_doc_node（顺序执行）

**触发条件**：Pipeline 启动后第一个执行。

**内部流程**：

```
init_doc_node(state)
  │
  ├─ MCP: check_version(operator_name, content_hash)
  │    └─ 查询数据库，返回 {status: "new"|"unchanged"|"updated", version, doc_id}
  │
  ├─ 如果 status == "unchanged"（内容完全未变）:
  │    ├─ MCP: get_parsed(operator_name, existing_version) → 取已有解析数据
  │    └─ 返回 {status="unchanged", doc_id, sections, cann_version, error=None}
  │       └─ 下游 _should_continue 会继续执行（仅在 error 时终止）
  │
  └─ 如果 status == "new" 或 "updated":
       ├─ MCP: save_doc(operator_name, content) → {operator_id, version, doc_id}
       ├─ MCP: parse_doc(content) → ParsedOperatorDocument（章节分割+分类）
       ├─ MCP: save_parsed(operator_name, version, parsed_dict) → 持久化解析结果
       └─ 返回 {status, version, doc_id, cann_version, sections（摘要）, error=None}
```

**输出状态**：`operator_id`, `doc_id`, `version`, `sections`, `cann_version`, `status`, `error`

---

#### 路由函数：_should_continue

```python
def _should_continue(state: dict) -> list[str]:
    if state.get("status") in ("error",):
        return END                    # 终止 Pipeline
    return ["product_support", "parse_params"]  # 扇出到两个并行节点
```

**关键**：`END` 是 LangGraph 内置常量，表示图终止。

---

#### 节点2a: product_support_node（与 2b 并行执行）

**触发条件**：init_doc 完成后，与 parse_params_node 同时启动。

**内部流程**：

```
product_support_node(state)
  │
  ├─ MCP: get_parsed_by_doc_id(doc_id) → 完整解析文档
  │
  ├─ 在 sections 中查找 section_type == "product_support"
  │    └─ 如未找到，直接返回空结果
  │
  ├─ LLM: ChatOpenAI(product_support_content, PRODUCT_SUPPORT_EXTRACT_PROMPT)
  │    └─ 温度 0.1，使用 settings 中配置的活跃 LLM 提供商
  │    └─ 提取 [{product: str, supported: bool}]
  │
  └─ MCP: save_product_support(doc_id, products)
       └─ 存到 document_versions.product_support 列
```

**输出状态**：`product_support: list[dict]`, `error`

---

#### 节点2b: parse_params_node（与 2a 并行执行）

**触发条件**：init_doc 完成后，与 product_support_node 同时启动。

**内部流程**：

```
parse_params_node(state)
  │
  ├─ MCP: get_parsed_by_doc_id(doc_id) → 完整解析文档
  │
  ├─ 在 sections 中查找 section_type == "function_prototype"
  │    └─ 如未找到，直接返回空结果
  │
  ├─ LLM: ChatOpenAI(func_proto_content, _EXTRACT_PROMPT)
  │    └─ 温度 0.1，使用活跃 LLM 提供商
  │    └─ 提取 [{function: str, parameter: [str]}]
  │
  ├─ _flatten_to_parameters(functions)
  │    └─ 每个函数-参数对 → 一条记录
  │    └─ 推断方向：参数名含 "out" → output，否则 → input
  │
  └─ MCP: save_parameters(doc_id, parameters)
       └─ INSERT OR REPLACE 到 parameters 表
```

**输出状态**：`parameters: list[dict]`, `error`

---

#### 节点3: param_desc_extract_node（顺序执行，等待 2a 和 2b 都完成）

**触发条件**：product_support_node 和 parse_params_node **都完成后**才执行（扇入汇合）。

**内部流程**：

```
param_desc_extract_node(state)
  │
  ├─ MCP: query_params_by_doc_id(doc_id) → 刚保存的参数列表
  │    └─ 如无参数，直接返回
  │
  ├─ MCP: get_section(doc_id, "params_get_workspace") → 参数说明章节内容
  │    └─ 如无内容，直接返回
  │
  ├─ 并发 LLM 调用（asyncio.Semaphore(5)）:
  │    └─ 对每个参数并行调用（最多 5 个并发）:
  │       LLM: ChatOpenAI(PARAM_DESC_EXTRACT_PROMPT.format(param_name, content))
  │       └─ 温度 0.1，提取该参数的描述文本
  │
  └─ MCP: update_param_descriptions(doc_id, updates)
       └─ 批量 UPDATE parameters 表的 description 等字段
```

**输出状态**：`error`

**注意**：当前实现中 `_extract_one()` 将 `usage_notes`, `dtype_desc`, `dformat_desc`, `shape`, `memory_desc` 全部硬编码为空字符串，**会覆盖已有数据**。

---

### 5.3 执行时序总结

```
时间轴 →

t0:  init_doc ──────────────────────┐
                                     │ (扇出)
t1:               ┌─────────────────┴──────────────────┐
                  ▼                                     ▼
t2:        product_support                      parse_params
           (LLM 提取产品表)                     (LLM 提取参数列表)
                  │                                     │
t3:               └─────────────────┬──────────────────┘
                                     │ (扇入 — 等待两者都完成)
t4:                                 ▼
                           param_desc_extract
                           (LLM 并发提取每个参数的描述)
                                     │
t5:                                 ▼
                                   END
```

**MCP 通信模式**：每个节点每次调用 MCP 工具时，`MCPClient._call_tool()` 会**启动一个新的 MCP Server 子进程**（`async with stdio_client`），调用完成后进程退出。整个 Pipeline 执行一次约产生 10+ 次子进程启动。

---

## 6. 数据库 Schema

```sql
-- 算子注册表
operators (
    id INTEGER PK,
    name TEXT NOT NULL UNIQUE,
    source_url TEXT,
    created_at TEXT
)

-- 文档版本表（每次上传新版本追加一行）
document_versions (
    id INTEGER PK,
    operator_id INTEGER FK → operators(id),
    version INTEGER NOT NULL,
    content TEXT NOT NULL,        -- 原始 Markdown 全文
    content_hash TEXT NOT NULL,   -- SHA256
    parsed_data TEXT,             -- JSON: ParsedOperatorDocument
    product_support TEXT,         -- JSON: [{product, supported}]
    created_at TEXT,
    UNIQUE(operator_id, version)
)

-- 参数表
parameters (
    id INTEGER PK,
    doc_id INTEGER FK → document_versions(id),
    function_name TEXT NOT NULL,
    param_name TEXT NOT NULL,
    param_type TEXT DEFAULT '',
    direction TEXT DEFAULT 'input',  -- 'input' | 'output'
    description TEXT,                -- Markdown 格式
    usage_notes TEXT,
    dtype_desc TEXT,
    dformat_desc TEXT,
    shape TEXT,
    memory_desc TEXT,
    created_at TEXT,
    UNIQUE(doc_id, function_name, param_name)
)
```

---

## 7. API 接口

| 方法 | 路径 | 作用 | 流程 |
|------|------|------|------|
| GET | `/` | 返回 static/index.html 测试页面 | 静态文件 |
| GET | `/health` | 健康检查 | 直接返回 `{"status": "ok"}` |
| POST | `/api/v1/upload` | 上传算子文档，触发完整 Pipeline | init_doc → [product_support ∥ parse_params] → param_desc_extract |
| GET | `/api/v1/operators` | 列出所有已注册算子及最新版本 | MCP: query_operators |
| GET | `/api/v1/operators/{name}` | 获取算子的完整解析数据 | MCP: get_parsed(name, ?version) |
| GET | `/api/v1/parameters` | 查询参数列表（可选 ?operator_name 过滤） | MCP: query_params(operator_name?) |

---

## 8. 配置项 (.env)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `zai` | LLM 提供商：`zai` 或 `deepseek` |
| `LLM_TEMPERATURE` | `0.7` | LLM 温度（全局默认，节点内覆盖为 0.1） |
| `ZAI_API_KEY` | (空) | Z.AI GLM API Key |
| `ZAI_BASE_URL` | `https://api.z.ai/api/paas/v4/` | Z.AI 端点 |
| `ZAI_MODEL` | `glm-5.1` | Z.AI 模型名 |
| `DEEPSEEK_API_KEY` | (空) | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek 端点 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | DeepSeek 模型名 |
| `DATABASE_PATH` | `data/operator_agent.db` | SQLite 数据库路径 |
| `OPERATORS_DIR` | `operators` | 算子文档目录 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `DEBUG` | `false` | 调试模式 |

---

## 9. 已知问题与待改进项

### 性能
1. **每次工具调用启动新进程**：`MCPClient._call_tool()` 每次创建新的 MCP Server 子进程，无连接复用，频繁启动开销大
2. **同步 SQLite**：MCP Server 全部使用同步 SQLite（`aiosqlite` 在依赖中声明但未使用）

### 代码质量
3. **`_parse_json_response` 重复**：`parse_params.py` 和 `product_support.py` 中相同函数各一份
4. **`param_desc_extract_node` 覆盖问题**：`_extract_one()` 将 description 之外的字段硬编码为空字符串，每次更新会清除已有数据

### 安全
5. **`.env` 含真实 API Key**：DeepSeek 密钥被写入 `.env` 文件（建议轮换并使用 Secret Manager）
6. **未集成 bandit 扫描**：CLAUDE.md 要求 `bandit -r src/`，但未配置

### 测试
7. **测试覆盖严重不足**：仅 `mcp_server/parsers/` 有 25 个测试，agent 包全部无测试
8. **无 CI/CD**：没有 GitHub Actions、Dockerfile 或部署配置

### 文档/结构
9. **CLAUDE.md 架构图过时**：描述 `src/agent/`、`src/api/` 扁平结构，实际为 `packages/*/src/` monorepo
10. **LangGraph Tools 存根**：`agent/tools/` 下的 `@tool` 定义未被节点使用（节点直接调 MCPClient）

### 前端
11. **字段名不匹配**：`GET /api/v1/parameters` 返回 `dtype_desc`/`dformat_desc`，但前端读取 `p.data_type`/`p.data_format`，导致这两列始终显示 `-`

---

## 10. 架构设计文档引用

| 文档 | 路径 | 作用 |
|------|------|------|
| 系统设计 | `docs/design.md` | 完整系统设计，含 12 节点 Pipeline 规划 |
| 实现计划 | `docs/plan.md` | 分阶段实现计划 v1.0 |
| 文档结构 | `docs/operator_doc_struct.md` | CANN 算子文档结构分析 |
| 提取指南 | `docs/prompts/*.md` | 各阶段 LLM 提取提示模板（7 个） |
