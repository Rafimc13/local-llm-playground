"""Shared runner types for external inference engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.schemas.model import BackendName, Quantization


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Memory and scheduling options used to launch one engine."""

    name: str
    context_length: int
    max_running_requests: int
    gpu_memory_utilization: float
    cpu_offload_gb: float
    chunked_prefill_size: int | None
    enforce_eager: bool


class EngineRunner(ABC):
    """Build reproducible commands without coupling them to HTTP inference."""

    backend: BackendName

    @abstractmethod
    def build_command(
        self,
        checkpoint: str,
        profile: RuntimeProfile,
        quantization: Quantization,
        port: int,
    ) -> tuple[str, ...]:
        """Build the argv tuple used to start an engine process."""
