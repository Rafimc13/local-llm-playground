"""vLLM backend adapter."""

from src.core.infrastructure.backends.openai_compatible import OpenAICompatibleBackend
from src.core.schemas.model import BackendName


class VLLMBackend(OpenAICompatibleBackend):
    """Connect to a vLLM OpenAI-compatible server."""

    name = BackendName.VLLM
