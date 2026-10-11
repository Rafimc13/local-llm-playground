"""SGLang backend adapter."""

from src.core.infrastructure.backends.openai_compatible import OpenAICompatibleBackend
from src.core.schemas.model import BackendName


class SGLangBackend(OpenAICompatibleBackend):
    """Connect to an SGLang OpenAI-compatible server."""

    name = BackendName.SGLANG
