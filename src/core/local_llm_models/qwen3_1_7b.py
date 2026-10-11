"""Qwen3 1.7B model specification."""

from src.core.local_llm_models.base import ModelSpec
from src.core.schemas.model import BackendName, Quantization

QWEN3_1_7B = ModelSpec(
    id="qwen3-1.7b",
    display_name="Qwen3 1.7B",
    parameter_count_billions=1.7,
    backend_ids={
        BackendName.OLLAMA: "qwen3:1.7b-q4_K_M",
        BackendName.VLLM: "Qwen/Qwen3-1.7B",
        BackendName.SGLANG: "Qwen/Qwen3-1.7B",
    },
    supported_quantizations={
        BackendName.OLLAMA: (Quantization.Q4_K_M, Quantization.Q8),
        BackendName.VLLM: (Quantization.FP16,),
        BackendName.SGLANG: (Quantization.FP16,),
    },
)
