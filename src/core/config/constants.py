"""Shared application constants."""

from typing import Final

API_PREFIX: Final = "/api/v1/local-llm"
DEFAULT_CONTEXT_LENGTH: Final = 2048
DEFAULT_MAX_TOKENS: Final = 256
DEFAULT_TEMPERATURE: Final = 0.7
SUPPORTED_BACKENDS: Final = ("ollama", "vllm", "sglang")
