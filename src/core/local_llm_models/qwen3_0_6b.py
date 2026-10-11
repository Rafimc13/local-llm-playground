"""Qwen3 0.6B model specification."""

from src.core.local_llm_models.base import ModelSpec
from src.core.schemas.model import BackendName, Quantization

QWEN3_0_6B = ModelSpec(
    id="qwen3-0.6b",
    display_name="Qwen3 0.6B",
    parameter_count_billions=0.6,
    backend_ids={
        BackendName.OLLAMA: "qwen3:0.6b-q4_K_M",
        BackendName.VLLM: "Qwen/Qwen3-0.6B",
        BackendName.SGLANG: "Qwen/Qwen3-0.6B",
    },
    supported_quantizations={
        BackendName.OLLAMA: (Quantization.Q4_K_M, Quantization.Q8),
        BackendName.VLLM: (Quantization.FP16,),
        BackendName.SGLANG: (Quantization.FP16,),
    },
)
