"""Quantization compatibility validation."""

from src.core.local_llm_models.base import ModelSpec
from src.core.schemas.model import BackendName, Quantization


def validate_quantization(
    model: ModelSpec,
    backend: BackendName,
    quantization: Quantization,
) -> None:
    """Reject unsupported checkpoint and quantization combinations."""
    supported = model.supported_quantizations.get(backend, ())
    if quantization not in supported:
        options = ", ".join(value.value for value in supported) or "none"
        message = (
            f"{model.id} does not support {quantization.value} on {backend.value}; "
            f"supported: {options}"
        )
        raise ValueError(message)
