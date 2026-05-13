---
name: mcp-server-patterns
description: Build MCP servers with Python SDK — tools, resources, prompts, Pydantic validation, stdio vs Streamable HTTP. Adapted for LangGraph + FastAPI + Python stack.
origin: ECC (adapted for Python)
---

# MCP Server Patterns (Python)

The Model Context Protocol (MCP) lets AI assistants call tools, read resources, and use prompts from your server. Use this skill when building or maintaining MCP servers with Python. The SDK API evolves; check the official MCP Python SDK docs for current method names and signatures.

## When to Use

Use when: implementing a new MCP server, adding tools or resources, choosing stdio vs HTTP, upgrading the SDK, or debugging MCP registration and transport issues.

## How It Works

### Core concepts

- **Tools**: Actions the model can invoke (e.g. search, query database, run a command). Register with `@server.tool()`.
- **Resources**: Read-only data the model can fetch (e.g. file contents, API responses, database records). Register with `@server.resource()`.
- **Prompts**: Reusable, parameterised prompt templates the client can surface. Register with `@server.prompt()`.
- **Transport**: stdio for local clients (e.g. Claude Desktop); Streamable HTTP for remote (Cursor, cloud deployments).

### Python MCP SDK

Install the official Python MCP SDK:

```bash
pip install mcp
# or with uv
uv add mcp
```

### Server setup with FastMCP

```python
from mcp.server.fastmcp import FastMCP

# Create server
mcp = FastMCP("my-server")

# Register a tool with Pydantic validation
@mcp.tool()
def search_documents(query: str, limit: int = 10) -> list[dict]:
    """Search documents by query string."""
    # Implementation here
    return results

# Register a resource
@mcp.resource("config://app")
def get_config() -> str:
    """Get application configuration."""
    return "configuration data"

# Register a prompt template
@mcp.prompt()
def review_code(code: str) -> str:
    """Generate a code review prompt."""
    return f"Please review the following code:\n\n{code}"
```

### Connecting with stdio

For local clients (Claude Desktop, Claude Code):

```python
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

# ... register tools, resources, prompts ...

if __name__ == "__main__":
    mcp.run()
```

Client configuration (`.claude.json` or `~/.claude.json`):

```json
{
  "mcpServers": {
    "my-server": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/project", "python", "-m", "my_server"]
    }
  }
}
```

### Transport: stdio vs HTTP

- **stdio**: Use for local development and Claude Desktop integration. Server reads from stdin, writes to stdout.
- **Streamable HTTP**: Use for remote/cloud deployments. Single HTTP endpoint per MCP spec.

```python
# stdio transport (default)
mcp.run()  # or mcp.run(transport="stdio")

# HTTP transport
mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

### Advanced: Tool with Pydantic models

```python
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(description="Search query string")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    filters: dict[str, str] | None = Field(default=None, description="Optional filters")

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    score: float

@mcp.tool()
def search_documents(request: SearchRequest) -> list[SearchResult]:
    """Search documents with filters and scoring."""
    # Implementation
    return results
```

### Integration with LangGraph

MCP servers can be used as LangGraph tools:

```python
from langchain_mcp import MCPToolkit

# Connect to MCP server
async with MCPToolkit(url="http://localhost:8000") as toolkit:
    tools = toolkit.get_tools()
    # Use tools in LangGraph agent
    agent = create_react_agent(model, tools)
```

Or expose LangGraph workflows as MCP tools:

```python
from my_graph import graph

@mcp.tool()
async def run_workflow(input_text: str) -> dict:
    """Run the LangGraph workflow on the given input."""
    result = await graph.ainvoke({"input": input_text})
    return result
```

### Integration with FastAPI

Mount MCP server alongside a FastAPI app:

```python
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP("my-server")

# Mount MCP on FastAPI
app.mount("/mcp", mcp.streamable_http_app())

# Regular FastAPI endpoints
@app.get("/health")
async def health():
    return {"status": "ok"}
```

## Best Practices

- **Schema first**: Define input schemas (Pydantic models) for every tool; document parameters and return shape.
- **Errors**: Return structured errors or messages the model can interpret; avoid raw stack traces.
- **Idempotency**: Prefer idempotent tools where possible so retries are safe.
- **Rate and cost**: For tools that call external APIs, consider rate limits and cost; document in the tool description.
- **Versioning**: Pin SDK version in pyproject.toml; check release notes when upgrading.
- **Separation**: Keep MCP tool definitions thin; delegate business logic to services.
- **Type safety**: Use Pydantic models for all tool inputs; leverage type annotations for outputs.

## Official SDKs and Docs

- **Python**: `mcp` package on PyPI (official Python SDK)
- **JavaScript/TypeScript**: `@modelcontextprotocol/sdk` (npm)
- **Go**: Official Go SDK on GitHub (`modelcontextprotocol/go-sdk`)
- **Docs**: https://modelcontextprotocol.io
