"""Model and backend API schemas."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class BackendName(StrEnum):
    """Supported inference engines."""

    OLLAMA = "ollama"
    VLLM = "vllm"
    SGLANG = "sglang"


class Quantization(StrEnum):
    """Checkpoint quantization families used by the playground."""

    FP16 = "fp16"
    Q4_K_M = "q4_k_m"
    Q8 = "q8"
    AWQ = "awq"
    GPTQ = "gptq"


class ModelInfo(BaseModel):
    """Serializable model capability description."""

    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    parameter_count_billions: float
    backend_ids: dict[BackendName, str]
    supported_quantizations: dict[BackendName, tuple[Quantization, ...]]
    default_context_length: int
    maximum_context_length: int
    recommended_max_tokens: int
    supports_thinking: bool = True


class BackendInfo(BaseModel):
    """Backend health and capability response."""

    name: BackendName
    base_url: str
    healthy: bool
    detail: str | None = None
    supports_streaming: bool = True
    supports_offloading: bool = False
    supports_continuous_batching: bool = False
