"""FastAPI and CLI dependency construction."""

from functools import lru_cache

from src.core.config.settings import get_settings
from src.core.infrastructure.backends.factory import BackendFactory
from src.core.local_llm_models.registry import model_registry
from src.services.local_llm.service import LocalLLMService


@lru_cache(maxsize=1)
def get_service() -> LocalLLMService:
    """Build the shared stateless application service."""
    return LocalLLMService(model_registry, BackendFactory(get_settings()))
