"""Central model registry used by the API and CLI."""

from collections.abc import Iterable

from src.core.local_llm_models.base import ModelSpec
from src.core.local_llm_models.qwen3_0_6b import QWEN3_0_6B
from src.core.local_llm_models.qwen3_1_7b import QWEN3_1_7B
from src.core.local_llm_models.qwen3_4b import QWEN3_4B
from src.core.schemas.model import BackendName


class ModelRegistry:
    """Look up model capabilities by stable playground identifier."""

    def __init__(self, models: Iterable[ModelSpec]) -> None:
        """Initialize the registry and require at least one model."""
        self._models = {model.id: model for model in models}
        if not self._models:
            msg = "At least one model must be registered"
            raise ValueError(msg)

    def list(self) -> tuple[ModelSpec, ...]:
        """Return registered models in deterministic order."""
        return tuple(self._models[key] for key in sorted(self._models))

    def get(self, model_id: str) -> ModelSpec:
        """Get a model or raise a user-facing validation error."""
        try:
            return self._models[model_id]
        except KeyError as error:
            available = ", ".join(sorted(self._models))
            message = f"Unknown model {model_id!r}. Available models: {available}"
            raise ValueError(message) from error

    def resolve_checkpoint(self, model_id: str, backend: BackendName) -> str:
        """Validate a model/backend combination and return its checkpoint."""
        return self.get(model_id).checkpoint_for(backend)


model_registry = ModelRegistry((QWEN3_0_6B, QWEN3_1_7B, QWEN3_4B))
