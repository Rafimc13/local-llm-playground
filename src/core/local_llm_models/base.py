"""Shared model specification."""

from dataclasses import dataclass

from src.core.schemas.model import BackendName, ModelInfo, Quantization


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable metadata describing one selectable model."""

    id: str
    display_name: str
    parameter_count_billions: float
    backend_ids: dict[BackendName, str]
    supported_quantizations: dict[BackendName, tuple[Quantization, ...]]
    default_context_length: int = 2048
    maximum_context_length: int = 40_960
    recommended_max_tokens: int = 512
    supports_thinking: bool = True

    def checkpoint_for(self, backend: BackendName) -> str:
        """Resolve the exact checkpoint name for an inference backend."""
        try:
            return self.backend_ids[backend]
        except KeyError as error:
            message = f"Model {self.id!r} does not support backend {backend.value!r}"
            raise ValueError(message) from error

    def to_info(self) -> ModelInfo:
        """Convert this domain object to an API schema."""
        return ModelInfo(
            id=self.id,
            display_name=self.display_name,
            parameter_count_billions=self.parameter_count_billions,
            backend_ids=self.backend_ids,
            supported_quantizations=self.supported_quantizations,
            default_context_length=self.default_context_length,
            maximum_context_length=self.maximum_context_length,
            recommended_max_tokens=self.recommended_max_tokens,
            supports_thinking=self.supports_thinking,
        )
