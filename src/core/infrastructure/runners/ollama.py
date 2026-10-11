"""Ollama runner configuration."""

from src.core.infrastructure.runners.base import EngineRunner, RuntimeProfile
from src.core.schemas.model import BackendName, Quantization


class OllamaRunner(EngineRunner):
    """Build an Ollama serve command; configuration is supplied via environment."""

    backend = BackendName.OLLAMA

    def build_command(
        self,
        checkpoint: str,
        profile: RuntimeProfile,
        quantization: Quantization,
        port: int,
    ) -> tuple[str, ...]:
        """Return the native Ollama server command."""
        del checkpoint, profile, quantization, port
        return ("ollama", "serve")

    @staticmethod
    def build_environment(profile: RuntimeProfile) -> dict[str, str]:
        """Build low-memory Ollama environment overrides."""
        return {
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": str(profile.max_running_requests),
            "OLLAMA_CONTEXT_LENGTH": str(profile.context_length),
        }
