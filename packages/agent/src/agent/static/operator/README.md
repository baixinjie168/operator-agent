# ⚠️ 这是前端用户 UI（不是后端代码）

本目录是 **operator-agent 的前端用户主操作台**（算子测试工作台），由历史项目 `operator-agent-ui` 于 2026-06-24 合并而来。

---

## 🚨 重要提示（请阅读后再修改）

| 项 | 说明 |
|---|---|
| **目录类型** | 前端静态资源（HTML / Markdown），**不是 Python 代码** |
| **同级的兄弟** | `../index.html` 才是管理面（系统健康检查 / 上传 / Debug），挂载在 `/`；本目录是用户主操作台，挂载在 `/operator/` |
| **后端在哪** | `packages/agent/src/agent/` 下的 Python 模块；本目录与后端无代码耦合，仅通过 `fetch('/api/v1/...')` 调用后端 API |
| **不要做的事** | ❌ 不要把 Python 模块、配置文件、`.env` 放进本目录  ❌ 不要把后端生成的 JSON 结果写回本目录  ❌ 不要修改 `index.html` 内的 `fetch("public/...")` 相对路径 |

---

## 📍 路由与挂载

由 [`packages/agent/src/agent/main.py`](../../../../main.py) 中如下代码挂载：

```python
operator_ui_dir = settings.static_dir / "operator"
if operator_ui_dir.exists():
    app.mount(
        "/operator",
        StaticFiles(directory=operator_ui_dir, html=True),
        name="operator-ui",
    )
```

| 访问路径 | 返回内容 |
|---|---|
| `GET /operator/` | 本目录 `index.html`（主操作台） |
| `GET /operator/public/docs/*.md` | 文档对比用的算子 Markdown |
| `GET /operator/public/prompts/*.md` | 约束 Agent 用的 prompt 模板 |

> **路径自洽**：`index.html` 内部 `fetch("public/...")` 是相对路径，浏览器会按当前 URL 自动解析为 `/operator/public/...`，**不需要改 `index.html` 里的任何路径**。

---

## 📁 当前内容（迁移清单）

```
operator/
├── README.md                                  ← 本文件
├── index.html                                 ← 主操作台前端（迁移自 operator-agent-ui/index.html）
└── public/
    ├── docs/
    │   ├── aclnnAddLayerNorm_v8.5.0.md         ← 文档对比（被 _fetchDocUrls 真实 fetch）
    │   └── aclnnAddLayerNorm_v9.0.0.md         ← 文档对比
    └── prompts/
        ├── functions_extracting_guide.md       ← 约束 Agent 提取 prompt（被 updateAgentCard 真实 fetch）
        └── functions_check_guide.md            ← 约束 Agent 校验 prompt
```

> **总共 5 个静态文件**。`operator-agent-ui/public/` 下的其它资源（`aclnn*_cases.json`、`aclnn*_extracted_constraints.json`、`constraints/*.json`、`validation/*.md`、`doc_struct/operator.json` 以及按算子分的 `aclnnAdaLayerNorm/`、`aclnnAdaptiveAvgPool3d/`、`aclnnAddLayerNorm/` 三个子目录）**没有迁移**——它们或被后端 API 替代、或属于死代码（参考 `opPath()` 在 `index.html` 中无调用方）。

---

## 🔧 修改指南

### 修改 UI 本身
直接编辑 `index.html`（仅一个文件）。修改后刷新 `/operator/` 即可看到效果，无需重启后端。

### 新增被 fetch 的静态资源
把文件放到 `public/<分类>/<文件名>`，然后在 `index.html` 里按 `fetch("public/<分类>/<文件名>")` 即可访问。无需修改后端。

### 新增后端 API
**不要**在本目录放任何后端代码。请在 `packages/agent/src/agent/routes/` 下加新 router，再在 `main.py` 里 `app.include_router(...)`。

### 旧项目清理
`z:\operator-test\operator-project\operator-agent-ui\` 目录迁移后已无业务代码（仅剩废弃的 `public/` 子资源）；后续如确认无历史价值，可整目录删除。

---

## 📜 历史

- 2026-06-24：从独立项目 `operator-agent-ui/` 迁入
- 路径决策：`static/operator/`（避免与 `static/index.html` 管理面混淆）
- 挂载点：`/operator/`
- 资源策略：仅迁移 `index.html` 真实 `fetch` 的 4 个 markdown + 1 个 `index.html`
