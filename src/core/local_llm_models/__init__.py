"""Model specifications and registry."""

from src.core.local_llm_models.base import ModelSpec
from src.core.local_llm_models.registry import ModelRegistry, model_registry

__all__ = ["ModelRegistry", "ModelSpec", "model_registry"]
