"""Server management routes: CRUD + SSH connection test."""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Optional

import asyncssh
from fastapi import APIRouter
from pydantic import BaseModel, Field

from agent.db import (
    create_server as db_create_server,
    delete_server as db_delete_server,
    get_server as db_get_server,
    query_servers as db_query_servers,
    update_server as db_update_server,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["servers"])


# ── Request / Response models ───────────────────────────────────────────────

class ServerCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    ip: str = Field(..., min_length=1)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    supported_product: str = Field(default="")


class ServerUpdateRequest(BaseModel):
    name: Optional[str] = None
    ip: Optional[str] = None
    port: Optional[int] = Field(default=None, ge=1, le=65535)
    username: Optional[str] = None
    password: Optional[str] = None
    supported_product: Optional[str] = None
    status: Optional[str] = None


# ── Routes ──────────────────────────────────────────────────────────────────

@router.get("/servers")
async def list_servers():
    """List all servers (newest first)."""
    servers = db_query_servers()
    return {"success": True, "servers": servers, "count": len(servers)}


@router.post("/servers")
async def add_server(body: ServerCreateRequest):
    """Add a new server."""
    try:
        server_id = db_create_server(
            name=body.name,
            ip=body.ip,
            port=body.port,
            username=body.username,
            password=body.password,
            supported_product=body.supported_product,
        )
        server = db_get_server(server_id)
        return {"success": True, "server": server}
    except Exception as e:
        logger.exception("Failed to create server")
        return {"success": False, "error": str(e)}


@router.put("/servers/{server_id}")
async def update_server_route(server_id: int, body: ServerUpdateRequest):
    """Update an existing server."""
    server = db_get_server(server_id)
    if not server:
        return {"success": False, "error": "Server not found"}
    fields = body.model_dump(exclude_none=True)
    if not fields:
        return {"success": False, "error": "No fields to update"}
    db_update_server(server_id, **fields)
    return {"success": True, "server": db_get_server(server_id)}


@router.delete("/servers/{server_id}")
async def delete_server_route(server_id: int):
    """Delete a server."""
    if not db_get_server(server_id):
        return {"success": False, "error": "Server not found"}
    db_delete_server(server_id)
    return {"success": True}


@router.post("/servers/{server_id}/test")
async def test_server_connection(server_id: int):
    """Test SSH connection to a server."""
    server = db_get_server(server_id)
    if not server:
        return {"success": False, "error": "Server not found"}

    result = await _test_ssh_connection(
        host=server["ip"],
        port=server["port"],
        username=server["username"],
        password=server["password"],
    )
    return result


@router.post("/servers/test")
async def test_connection_direct(body: ServerCreateRequest):
    """Test SSH connection without saving (for add/edit dialog preview)."""
    result = await _test_ssh_connection(
        host=body.ip,
        port=body.port,
        username=body.username,
        password=body.password,
    )
    return result


# ── SSH connection test ─────────────────────────────────────────────────────

async def _test_ssh_connection(
    host: str, port: int, username: str, password: str, timeout: float = 10.0
) -> dict:
    """Test SSH connection using asyncssh. Returns success/error dict."""
    # Step 1: TCP connectivity check
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        await asyncio.get_event_loop().run_in_executor(
            None, sock.connect, (host, port)
        )
        sock.close()
    except socket.timeout:
        return {"success": False, "error": f"连接超时: {host}:{port} 无响应"}
    except ConnectionRefusedError:
        return {"success": False, "error": f"连接被拒绝: {host}:{port}"}
    except OSError as e:
        return {"success": False, "error": f"网络错误: {e}"}

    # Step 2: SSH authentication check
    try:
        conn = await asyncio.wait_for(
            asyncssh.connect(
                host,
                port=port,
                username=username,
                password=password,
                known_hosts=None,
            ),
            timeout=timeout,
        )
        conn.close()
        return {
            "success": True,
            "message": f"SSH 连接成功: {username}@{host}:{port}",
        }
    except asyncssh.PermissionDenied:
        return {"success": False, "error": "认证失败: 用户名或密码错误"}
    except asyncssh.TimeoutError:
        return {"success": False, "error": "SSH 认证超时"}
    except Exception as e:
        logger.exception("SSH connection test failed")
        return {"success": False, "error": f"SSH 错误: {e}"}
