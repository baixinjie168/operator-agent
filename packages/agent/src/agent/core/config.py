from pathlib import Path

from pydantic import Field, SecretStr, model_validator
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

    # LLM configuration (Z.AI OpenAI-compatible API)
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.z.ai/api/paas/v4/"
    llm_model: str = "glm-5.1"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def _validate_llm_config(self) -> "Settings":
        if not self.llm_api_key.get_secret_value():
            raise ValueError("LLM_API_KEY is required. Set it in .env or as an environment variable.")
        return self


settings = Settings()
