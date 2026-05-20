from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_name: str = "operator-agent"
    debug: bool = False
    operators_dir: str = "operators"
    database_path: str = "data/operator_agent.db"
    log_level: str = "INFO"
    mcp_server_command: str = "python -m mcp_server"
    static_dir: Path = Path(__file__).resolve().parent.parent / "static"
    agent_model: str = "anthropic:claude-sonnet-4-20250514"


settings = Settings()
