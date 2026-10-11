"""vLLM launch-command builder."""

from src.core.infrastructure.runners.base import EngineRunner, RuntimeProfile
from src.core.schemas.model import BackendName, Quantization


class VLLMRunner(EngineRunner):
    """Build commands for the standalone vLLM server."""

    backend = BackendName.VLLM

    def build_command(
        self,
        checkpoint: str,
        profile: RuntimeProfile,
        quantization: Quantization,
        port: int,
    ) -> tuple[str, ...]:
        """Return a memory-conscious vLLM launch command."""
        command = [
            "vllm",
            "serve",
            checkpoint,
            "--max-model-len",
            str(profile.context_length),
            "--gpu-memory-utilization",
            str(profile.gpu_memory_utilization),
            "--max-num-seqs",
            str(profile.max_running_requests),
            "--max-num-batched-tokens",
            str(profile.context_length),
            "--port",
            str(port),
        ]
        if quantization is Quantization.FP16:
            command.extend(("--dtype", "float16"))
        elif quantization is Quantization.AWQ:
            command.extend(("--quantization", "awq"))
        if profile.cpu_offload_gb > 0:
            command.extend(("--cpu-offload-gb", str(profile.cpu_offload_gb)))
        if profile.enforce_eager:
            command.append("--enforce-eager")
        return tuple(command)
