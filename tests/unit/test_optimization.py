"""Optimization validation and runner tests."""

import pytest

from src.core.infrastructure.optimization.memory import available_vram_budget_mib
from src.core.infrastructure.optimization.quantization import validate_quantization
from src.core.infrastructure.runners.base import RuntimeProfile
from src.core.infrastructure.runners.vllm import VLLMRunner
from src.core.local_llm_models.registry import model_registry
from src.core.schemas.model import BackendName, Quantization


def test_vram_budget() -> None:
    assert abs(available_vram_budget_mib(6144, 0.8) - 4915.2) < 1e-9


def test_rejects_wrong_quantization() -> None:
    model = model_registry.get("qwen3-4b")
    with pytest.raises(ValueError, match="does not support fp16"):
        validate_quantization(model, BackendName.VLLM, Quantization.FP16)


def test_vllm_low_memory_command() -> None:
    profile = RuntimeProfile(
        name="low-memory",
        context_length=2048,
        max_running_requests=1,
        gpu_memory_utilization=0.8,
        cpu_offload_gb=0,
        chunked_prefill_size=1024,
        enforce_eager=True,
    )
    command = VLLMRunner().build_command("Qwen/Qwen3-0.6B", profile, Quantization.FP16, 8000)
    assert "--enforce-eager" in command
    assert command[:3] == ("vllm", "serve", "Qwen/Qwen3-0.6B")
