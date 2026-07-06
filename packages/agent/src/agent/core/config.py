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
    cases_dir: Path = Path(__file__).resolve().parents[5] / "cases"

    # Retry limits for LLM-based extraction (configurable per deployment)
    expr_max_retries: int = Field(default=2, ge=0, le=10)
    dimensions_max_retries: int = Field(default=2, ge=0, le=10)

    # Parallel execution: how many operators to parse concurrently within a task
    task_max_workers: int = Field(default=3, ge=1, le=20)
    task_max_retries: int = Field(default=1, ge=0, le=5)
    task_config_file: str = "config/task_config.yaml"
    semantic_rules_file: str = "config/semantic_value_rules.yaml"

    # Max tokens for LLM responses (prevents truncation of large JSON outputs)
    llm_max_tokens: int = Field(default=16384, ge=256, le=131072)

    # Layer 2 LLM switch for single-parameter constraint extraction.
    # When False (default), only Layer 1 deterministic regex rules run.
    enable_single_param_llm: bool = False

    # Force Phase 2b semantic verification for all relations (for analysis/debugging).
    # When True, Phase2b runs regardless of LLM confidence level.
    force_phase2b: bool = False

    # Phase 3 switches -------------------------------------------------------
    # constraint_semantic_dedup: cross-source / cross-expr_type semantic
    # dedup in constraint_extract (Item 7). When False, only the legacy
    # string-based _expr_exists dedup runs.
    constraint_semantic_dedup: bool = Field(default=True)
    # expr_simplify: post-generation simplification of over-long exprs in
    # complex_relation_agent + build_param_relations retry (Item 8). When
    # False, _simplify_expr is skipped.
    expr_simplify: bool = Field(default=True)

    # FIX-10: Source citation existence verification.
    # When True, constraints whose src_text cannot be found in the document
    # (ws + exe + constraints sections) are deleted as fabricated citations.
    # Complements FIX-2 (param legality) to cover "params legal but src_text
    # fabricated" scenarios.
    relation_verify_source: bool = Field(default=True)
    # Similarity threshold for the sliding-window fuzzy match (FIX-10b).
    # best >= threshold -> KEEP, otherwise DELETE. 0.3 is permissive on purpose
    # to avoid deleting legitimately-paraphrased citations.
    relation_verify_source_threshold: float = Field(default=0.3)

    # Constraint check Agent: separate model for post-pipeline verification.
    # Uses a different provider/model from the generation pipeline to avoid
    # self-evaluation bias.  Defaults to DeepSeek for stronger reasoning.
    constraint_check_llm_provider: LLMProvider = LLMProvider.DEEPSEEK
    constraint_check_model: str = ""  # empty = use provider's default model
    # Separate max_tokens for constraint check (HTML reports can be large).
    # Generation uses llm_max_tokens (default 16384), but constraint check
    # generates full HTML with CSS + tables + JS, which needs more tokens.
    constraint_check_max_tokens: int = Field(default=32768, ge=256, le=131072)

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

    def get_provider_api_key(self, provider: LLMProvider) -> str:
        """API key for a specific provider (not necessarily the active one)."""
        match provider:
            case LLMProvider.ZAI:
                return self.zai_api_key.get_secret_value()
            case LLMProvider.DEEPSEEK:
                return self.deepseek_api_key.get_secret_value()
            case LLMProvider.OWN_AI:
                return self.own_ai_api_key.get_secret_value()
            case LLMProvider.QWEN:
                return self.qwen_api_key.get_secret_value()

    def get_provider_base_url(self, provider: LLMProvider) -> str:
        """Base URL for a specific provider."""
        match provider:
            case LLMProvider.ZAI:
                return self.zai_base_url
            case LLMProvider.DEEPSEEK:
                return self.deepseek_base_url
            case LLMProvider.OWN_AI:
                return self.own_ai_base_url
            case LLMProvider.QWEN:
                return self.qwen_base_url

    def get_provider_model(self, provider: LLMProvider) -> str:
        """Default model name for a specific provider."""
        match provider:
            case LLMProvider.ZAI:
                return self.zai_model
            case LLMProvider.DEEPSEEK:
                return self.deepseek_model
            case LLMProvider.OWN_AI:
                return self.own_ai_model
            case LLMProvider.QWEN:
                return self.qwen_model


settings = Settings()
