"""FastAPI application factory for the operator-agent main system."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agent.core.config import settings
from agent.core.logging import setup_logging
from agent.db import get_db
from agent.routes.cases import router as cases_router
from agent.routes.chat import router as chat_router
from agent.routes.execute import router as execute_router
from agent.routes.generator import router as generator_router
from agent.routes.query import router as query_router
from agent.routes.runs import router as runs_router
from agent.routes.servers import router as servers_router
from agent.routes.task import router as task_router
from agent.routes.upload import router as upload_router
from agent.runtime import RuntimeManager


def create_app() -> FastAPI:
    setup_logging(settings.log_level)

    get_db()

    app = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        description="Operator Agent - CANN operator constraint extraction and test generation",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.runtime_manager = RuntimeManager()

    app.include_router(upload_router)
    app.include_router(query_router)
    app.include_router(runs_router)
    app.include_router(cases_router)
    app.include_router(generator_router)
    app.include_router(execute_router)
    app.include_router(chat_router)
    app.include_router(servers_router)
    app.include_router(task_router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (settings.static_dir / "index.html").read_text(encoding="utf-8")

    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    # 前端用户主操作台（迁移自 operator-agent-ui 项目，2026-06-24）
    # - 目录：packages/agent/src/agent/static/operator/
    # - 用途：算子测试工作台（用户主交互面），不是管理面
    # - 同级 index.html（/ 路由）才是管理面；请勿混淆
    # - 该目录结构见 static/operator/README.md
    operator_ui_dir = settings.static_dir / "operator"
    if operator_ui_dir.exists():
        app.mount(
            "/operator",
            StaticFiles(directory=operator_ui_dir, html=True),
            name="operator-ui",
        )

    return app
