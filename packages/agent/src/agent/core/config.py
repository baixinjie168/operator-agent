import sys
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.models.enums import LLMProvider

# Default configuration per provider
_PROVIDER_DEFAULTS: dict[LLMProvider, dict[str, str]] = {
    LLMProvider.ZAI: {
        "base_url": "https://api.z.ai/api/paas/v4/",
        "model": "glm-5.1",
    },
    LLMProvider.DEEPSEEK: {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    LLMProvider.OWN_AI: {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    LLMProvider.QWEN: {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-max",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_name: str = "operator-agent"
    debug: bool = False
    operators_dir: str = "operators"
    uploads_dir: str = "uploads"
    database_path: str = "data/operator_agent.db"
    log_level: str = "INFO"
    mcp_server_command: str = f"{sys.executable} -m mcp_server"
    static_dir: Path = Path(__file__).resolve().parent.parent / "static"

    # Retry limits for LLM-based extraction (configurable per deployment)
    expr_max_retries: int = Field(default=2, ge=0, le=10)
    dimensions_max_retries: int = Field(default=2, ge=0, le=10)

    # Layer 2 LLM switch for single-parameter constraint extraction.
    # When False (default), only Layer 1 deterministic regex rules run.
    enable_single_param_llm: bool = False

    # Master switch: which LLM provider to use
    llm_provider: LLMProvider = LLMProvider.ZAI

    # Z.AI-specific configuration
    zai_api_key: SecretStr = SecretStr("")
    zai_base_url: str = _PROVIDER_DEFAULTS[LLMProvider.ZAI]["base_url"]
    zai_model: str = _PROVIDER_DEFAULTS[LLMProvider.ZAI]["model"]

    # DeepSeek-specific configuration
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = _PROVIDER_DEFAULTS[LLMProvider.DEEPSEEK]["base_url"]
    deepseek_model: str = _PROVIDER_DEFAULTS[LLMProvider.DEEPSEEK]["model"]

    # Own-AI-specific configuration (OpenAI-compatible)
    own_ai_api_key: SecretStr = SecretStr("")
    own_ai_base_url: str = _PROVIDER_DEFAULTS[LLMProvider.OWN_AI]["base_url"]
    own_ai_model: str = _PROVIDER_DEFAULTS[LLMProvider.OWN_AI]["model"]

    # Qwen-specific configuration (DashScope OpenAI-compatible)
    qwen_api_key: SecretStr = SecretStr("")
    qwen_base_url: str = _PROVIDER_DEFAULTS[LLMProvider.QWEN]["base_url"]
    qwen_model: str = _PROVIDER_DEFAULTS[LLMProvider.QWEN]["model"]

    # Shared LLM settings
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def _validate_llm_config(self) -> "Settings":
        api_key = self._active_api_key()
        if not api_key:
            raise ValueError(
                f"API key for provider '{self.llm_provider}' is required. "
                f"Set {self.llm_provider.value.upper()}_API_KEY in .env or as an environment variable."
            )
        return self

    @property
    def active_api_key(self) -> SecretStr:
        """API key for the active provider."""
        return SecretStr(self._active_api_key())

    @property
    def active_base_url(self) -> str:
        """Base URL for the active provider."""
        match self.llm_provider:
            case LLMProvider.ZAI:
                return self.zai_base_url
            case LLMProvider.DEEPSEEK:
                return self.deepseek_base_url
            case LLMProvider.OWN_AI:
                return self.own_ai_base_url
            case LLMProvider.QWEN:
                return self.qwen_base_url

    @property
    def active_model(self) -> str:
        """Model name for the active provider."""
        match self.llm_provider:
            case LLMProvider.ZAI:
                return self.zai_model
            case LLMProvider.DEEPSEEK:
                return self.deepseek_model
            case LLMProvider.OWN_AI:
                return self.own_ai_model
            case LLMProvider.QWEN:
                return self.qwen_model

    def _active_api_key(self) -> str:
        match self.llm_provider:
            case LLMProvider.ZAI:
                return self.zai_api_key.get_secret_value()
            case LLMProvider.DEEPSEEK:
                return self.deepseek_api_key.get_secret_value()
            case LLMProvider.OWN_AI:
                return self.own_ai_api_key.get_secret_value()
            case LLMProvider.QWEN:
                return self.qwen_api_key.get_secret_value()


settings = Settings()
