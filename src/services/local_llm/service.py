"""Application service shared by REST and CLI interfaces."""

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from src.core.benchmarking.runner import BenchmarkRunner
from src.core.infrastructure.backends.base import LLMBackend
from src.core.infrastructure.monitoring.gpu_metrics import GPUMetrics, read_gpu_metrics
from src.core.local_llm_models.registry import ModelRegistry
from src.core.schemas.benchmark import BenchmarkRequest, BenchmarkResult
from src.core.schemas.chat import ChatRequest, ChatResponse
from src.core.schemas.model import BackendInfo, BackendName, ModelInfo


class BackendProvider(Protocol):
    """Structural interface for production and test backend factories."""

    def create(self, name: BackendName) -> LLMBackend:
        """Create one backend adapter."""
        ...


class LocalLLMService:
    """Coordinate model validation, backends, monitoring, and benchmarks."""

    def __init__(self, registry: ModelRegistry, backend_factory: BackendProvider) -> None:
        """Initialize with model and backend providers."""
        self._registry = registry
        self._backend_factory = backend_factory

    def list_models(self) -> tuple[ModelInfo, ...]:
        """List every registered model and supported configuration."""
        return tuple(model.to_info() for model in self._registry.list())

    async def list_backends(self) -> tuple[BackendInfo, ...]:
        """Probe all inference servers concurrently without loading models."""
        return tuple(
            await asyncio.gather(
                *(self._backend_factory.create(name).health() for name in BackendName)
            )
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Validate and dispatch one complete chat request."""
        checkpoint = self._registry.resolve_checkpoint(request.model, request.backend)
        backend = self._backend_factory.create(request.backend)
        return await backend.chat(request, checkpoint)

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Validate and dispatch one streaming chat request."""
        checkpoint = self._registry.resolve_checkpoint(request.model, request.backend)
        backend = self._backend_factory.create(request.backend)
        async for fragment in backend.stream(request, checkpoint):
            yield fragment

    async def benchmark(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Run a repeatable benchmark using normal service inference."""
        checkpoint = self._registry.resolve_checkpoint(request.model, request.backend)
        runner = BenchmarkRunner(self.chat)
        return await runner.run(request, checkpoint)

    @staticmethod
    def gpu_metrics() -> tuple[GPUMetrics, ...]:
        """Read current GPU telemetry without starting inference."""
        return read_gpu_metrics()
