from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigSetting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENROUTER_API_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str
    LANGFUSE_MEDIA_UPLOAD_ENABLED: bool = False
    TEAM_NAME: str

    LOCAL_DEV_NO_TRACING: bool = False
    USE_MOCK_LLM: bool = False
    USE_LOCAL_LLM: bool = False
    LOCAL_LLM_URL: str = "http://localhost:11434/v1"
    LOCAL_LLM_MODEL: str = "qwen2.5:14b"
    APP_ENV: str = "dev"

    MODEL_ID: str = "gpt-4o-mini"
    MODEL_TEMPERATURE: float = 0.7
    MODEL_MAX_TOKENS: int = Field(default=1000, gt=0)

    DISPATCHER_MODEL_ID: str = "openai/gpt-4o-mini"
    SPECIALIST_MODEL_ID: str = "openai/gpt-4o-mini"
    JUDGE_MODEL_ID: str = "anthropic/claude-3.7-sonnet"
    DETECTIVE_MODEL_ID: str = "deepseek/deepseek-r1"

    AGENT_TOOL_TRANSPORT: str = "mcp"
    AGENT_MCP_STARTUP_TIMEOUT_SECONDS: int = Field(default=8, ge=1)
    AGENT_MAX_TOOL_RETRIES: int = Field(default=2, ge=0)
    AGENT_MAX_GRAPH_STEPS: int = Field(default=12, ge=4)
    AGENT_SQL_MAX_ROWS: int = Field(default=50, ge=1)
    AGENT_PANDAS_MAX_ROWS: int = Field(default=50, ge=1)

    TEST_ROW_LIMIT: int | None = None

    @model_validator(mode="after")
    def validate_llm_mode(self) -> "ConfigSetting":
        if self.USE_MOCK_LLM and self.USE_LOCAL_LLM:
            raise ValueError("USE_MOCK_LLM and USE_LOCAL_LLM cannot both be true.")
        return self
    
@lru_cache
def get_config() -> ConfigSetting:
    return ConfigSetting()


config = get_config()
