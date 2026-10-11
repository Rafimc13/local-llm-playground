"""Inference backend construction."""

from src.core.config.settings import Settings
from src.core.infrastructure.backends.base import LLMBackend
from src.core.infrastructure.backends.ollama import OllamaBackend
from src.core.infrastructure.backends.sglang import SGLangBackend
from src.core.infrastructure.backends.vllm import VLLMBackend
from src.core.schemas.model import BackendName


class BackendFactory:
    """Create backend adapters from environment settings."""

    def __init__(self, settings: Settings) -> None:
        """Initialize from validated application settings."""
        self._settings = settings

    def create(self, name: BackendName) -> LLMBackend:
        """Create the requested stateless HTTP adapter."""
        timeout = self._settings.request_timeout_seconds
        if name is BackendName.OLLAMA:
            return OllamaBackend(str(self._settings.ollama_base_url), timeout)
        if name is BackendName.VLLM:
            return VLLMBackend(str(self._settings.vllm_base_url), timeout)
        if name is BackendName.SGLANG:
            return SGLangBackend(str(self._settings.sglang_base_url), timeout)
        msg = f"Unsupported backend: {name}"
        raise ValueError(msg)
