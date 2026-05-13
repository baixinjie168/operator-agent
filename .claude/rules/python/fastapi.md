---
paths:
  - "**/app/**/*.py"
  - "**/fastapi/**/*.py"
  - "**/*_api.py"
---
# FastAPI Rules

Use these rules for FastAPI projects alongside the general Python rules.

## Structure

- Put app construction in `create_app()`.
- Keep routers thin; move persistence and business behavior into services or CRUD helpers.
- Keep request schemas, update schemas, and response schemas separate.
- Keep database sessions and auth in dependencies.

## Async

- Use `async def` for endpoints that perform I/O.
- Use async database and HTTP clients from async endpoints.
- Do not call `requests`, sync SQLAlchemy sessions, or blocking file/network operations from async routes.

## Dependency Injection

```python
@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ...
```

Do not create `SessionLocal()` or long-lived clients inside route handlers.

## Schemas

- Never include passwords, password hashes, access tokens, refresh tokens, or internal auth state in response models.
- Use `response_model` on endpoints that return application data.
- Use field constraints instead of hand-written validation when Pydantic can express the rule.

## Security

- Keep CORS origins environment-specific.
- Do not combine wildcard origins with credentialed CORS.
- Validate JWT expiry, issuer, audience, and algorithm.
- Rate-limit auth and write-heavy endpoints.
- Redact credentials, cookies, authorization headers, and tokens from logs.

## Testing

- Override the exact dependency used by `Depends`.
- Clear `app.dependency_overrides` after tests.
- Prefer async test clients for async applications.

## LangGraph Integration

- Define graph state with `TypedDict` or `PydanticModel`.
- Keep nodes as pure functions: receive state, return partial state update.
- Use `RunnableConfig` for passing metadata, not the state dict.
- Register LangGraph tools as MCP tools when exposing via MCP server.

## MCP Server Integration

- Mount MCP server on FastAPI via `app.mount("/mcp", mcp.streamable_http_app())`.
- Keep MCP tool handlers thin; delegate to service layer.
- Use Pydantic models for MCP tool input schemas.

See skill: `mcp-server-patterns`.
