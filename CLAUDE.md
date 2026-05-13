# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

A **LangGraph-based intelligent agent** for task automation, combining custom Tools and MCP servers (FastAPI backend). Built with Python, following production-grade conventions from the ECC (Everything Claude Code) component system.

**Stack**: Python 3.12+, LangGraph, FastAPI, MCP Python SDK, Pydantic, pytest, uv

## Architecture

```
src/
├── agent/              # LangGraph graph definitions, nodes, state schemas
│   ├── graph.py        # Main workflow graph
│   ├── nodes/          # Individual graph nodes (pure functions)
│   ├── state.py        # TypedDict / Pydantic state definitions
│   └── tools/          # Custom LangGraph tools
├── mcp_server/         # MCP server implementation
│   ├── server.py       # FastMCP server with tool/resource/prompt registration
│   ├── tools/          # MCP tool handlers
│   └── resources/      # MCP resource handlers
├── api/                # FastAPI application
│   ├── app.py          # create_app(), router mounting
│   ├── routes/         # API route handlers (thin)
│   ├── schemas/        # Request/response Pydantic models
│   └── dependencies.py # FastAPI Depends() providers
├── services/           # Business logic (shared by API, MCP, and agent)
├── models/             # Domain models and data access
└── core/               # Config, exceptions, logging, middleware
```

## Python Conventions

- **Type hints** on all function signatures; use `mypy --strict` or `pyright`.
- **Pydantic v2** for all schemas and validation.
- **Immutability**: prefer `dataclass(frozen=True)`, `NamedTuple`, or Pydantic models over mutable dicts.
- **Error handling**: specific exceptions (no bare `except`), exception chaining with `raise ... from e`.
- **Async**: `async def` for all I/O-bound operations (API endpoints, MCP tools, LangGraph nodes with I/O).
- **Logging**: `logging.getLogger(__name__)` — never `print()`.
- **Imports**: stdlib → third-party → local; use `isort` (enforced by ruff).
- **Formatting**: `black` (line length 120); linting: `ruff`.

## LangGraph Patterns

- **State**: Define graph state with `TypedDict` or Pydantic model; keep fields flat and explicit.
- **Nodes**: Pure functions that receive state and return a partial state update dict.
- **Edges**: Use conditional edges for branching logic; prefer explicit over implicit routing.
- **Tools**: Register as `@tool` decorated functions with clear docstrings for the LLM.
- **Testing**: Test nodes in isolation with mock state; test graph integration with full state flow.

## MCP Server Patterns

- Use `FastMCP` from the `mcp` Python package.
- Register tools with `@mcp.tool()`, resources with `@mcp.resource()`, prompts with `@mcp.prompt()`.
- Validate all tool inputs with Pydantic models.
- Keep tool handlers thin; delegate to the service layer.
- Use stdio transport for local development, Streamable HTTP for remote.

## FastAPI Patterns

- App construction in `create_app()`.
- Thin routers; business logic in services.
- Separate request, update, and response schemas.
- `Depends()` for DB sessions and auth — never create `SessionLocal()` in route handlers.
- Async test client for async apps; clear `app.dependency_overrides` after tests.

## Testing

- **Framework**: pytest with pytest-asyncio for async tests.
- **Coverage**: 80% minimum; use `pytest --cov=src --cov-report=term-missing`.
- **TDD**: Write failing test first (RED) → minimal implementation (GREEN) → refactor (IMPROVE).
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`.
- **Fixtures**: Shared fixtures in `conftest.py`; use `tmp_path` for file tests.
- **Mocking**: `@patch` for external dependencies; mock MCP tool calls in agent tests.

## Tooling

```bash
# Lint and format
ruff check . --fix
black .

# Type check
mypy . --strict

# Test
pytest --cov=src -v

# Run MCP server locally
uv run python -m src.mcp_server.server

# Run FastAPI app
uvicorn src.api.app:create_app --factory --reload
```

## Available Commands

| Command | Purpose |
|---------|---------|
| `/plan` | Implementation planning with risk assessment |
| `/code-review` | Quality and security review |
| `/build-fix` | Fix build/type/lint errors |
| `/quality-gate` | Run verification pipeline |
| `/python-review` | Python-specific code review |
| `/checkpoint` | Save/verify workflow checkpoints |
| `/skill-create` | Extract patterns from git history |

## Available Agents

| Agent | When to Use |
|-------|-------------|
| `planner` | Complex feature/refactoring planning |
| `architect` | System design and architecture decisions |
| `tdd-guide` | Test-driven development workflow |
| `python-reviewer` | Python code review (PEP 8, types, security) |
| `code-reviewer` | General code quality review |
| `security-reviewer` | Security vulnerability analysis (OWASP) |
| `build-error-resolver` | Fix build/type/lint errors |
| `refactor-cleaner` | Dead code cleanup |

## Security

- No hardcoded secrets; use environment variables or `.env` with `python-dotenv`.
- Parameterized queries only — never concatenate SQL.
- Validate all external inputs with Pydantic schemas.
- Rate-limit auth and write-heavy endpoints.
- CORS origins must be environment-specific.
- Run `bandit -r src/` before commits.

## Git Workflow

- **Commits**: `<type>: <description>` (feat, fix, refactor, docs, test, chore, perf, ci).
- **Branches**: Feature branches from main; PRs require passing tests.
- **Never commit** `.env`, secrets, or credentials.
