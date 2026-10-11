"""Environment-backed application settings."""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the orchestration service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LOCAL_LLM_",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:11434")
    vllm_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8000/v1")
    sglang_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:30000/v1")
    default_model: str = "qwen3-0.6b"
    default_backend: Literal["ollama", "vllm", "sglang"] = "ollama"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
