"""Qwen3 4B model specification."""

from src.core.local_llm_models.base import ModelSpec
from src.core.schemas.model import BackendName, Quantization

QWEN3_4B = ModelSpec(
    id="qwen3-4b",
    display_name="Qwen3 4B",
    parameter_count_billions=4.0,
    backend_ids={
        BackendName.OLLAMA: "qwen3:4b-q4_K_M",
        BackendName.VLLM: "Qwen/Qwen3-4B-AWQ",
        BackendName.SGLANG: "Qwen/Qwen3-4B-AWQ",
    },
    supported_quantizations={
        BackendName.OLLAMA: (Quantization.Q4_K_M, Quantization.Q8),
        BackendName.VLLM: (Quantization.AWQ,),
        BackendName.SGLANG: (Quantization.AWQ,),
    },
)
