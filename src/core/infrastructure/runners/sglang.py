"""SGLang launch-command builder."""

from src.core.infrastructure.runners.base import EngineRunner, RuntimeProfile
from src.core.schemas.model import BackendName, Quantization


class SGLangRunner(EngineRunner):
    """Build commands for the standalone SGLang server."""

    backend = BackendName.SGLANG

    def build_command(
        self,
        checkpoint: str,
        profile: RuntimeProfile,
        quantization: Quantization,
        port: int,
    ) -> tuple[str, ...]:
        """Return a memory-conscious SGLang launch command."""
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            checkpoint,
            "--context-length",
            str(profile.context_length),
            "--mem-fraction-static",
            str(profile.gpu_memory_utilization),
            "--max-running-requests",
            str(profile.max_running_requests),
            "--port",
            str(port),
        ]
        if quantization is Quantization.FP16:
            command.extend(("--dtype", "float16"))
        elif quantization is Quantization.AWQ:
            command.extend(("--quantization", "awq"))
        if profile.chunked_prefill_size:
            command.extend(("--chunked-prefill-size", str(profile.chunked_prefill_size)))
        if profile.cpu_offload_gb > 0:
            command.extend(("--cpu-offload-gb", str(profile.cpu_offload_gb)))
        if profile.enforce_eager:
            command.append("--disable-cuda-graph")
        return tuple(command)
