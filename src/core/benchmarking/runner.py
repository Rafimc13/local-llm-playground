"""Benchmark runner using the same chat path as normal inference."""

from collections.abc import Awaitable, Callable

from src.core.benchmarking.metrics import mean_latency, mean_tokens_per_second
from src.core.schemas.benchmark import BenchmarkRequest, BenchmarkResult, BenchmarkSample
from src.core.schemas.chat import ChatRequest, ChatResponse

ChatCallable = Callable[[ChatRequest], Awaitable[ChatResponse]]


class BenchmarkRunner:
    """Execute warm and measured runs over a fixed prompt collection."""

    def __init__(self, chat: ChatCallable) -> None:
        """Initialize with the application chat callable under measurement."""
        self._chat = chat

    async def run(self, request: BenchmarkRequest, checkpoint: str) -> BenchmarkResult:
        """Run the configured benchmark sequentially for a stable baseline."""
        for prompt in request.prompts:
            warmup_request = self._chat_request(request, prompt)
            for _ in range(request.warmup_runs):
                _ = await self._chat(warmup_request)

        samples: list[BenchmarkSample] = []
        for prompt_index, prompt in enumerate(request.prompts):
            chat_request = self._chat_request(request, prompt)
            for run_index in range(request.measured_runs):
                try:
                    response = await self._chat(chat_request)
                    samples.append(
                        BenchmarkSample(
                            prompt_index=prompt_index,
                            run_index=run_index,
                            latency_seconds=response.latency_seconds,
                            completion_tokens=response.completion_tokens,
                            tokens_per_second=response.tokens_per_second,
                        )
                    )
                except RuntimeError as error:
                    samples.append(
                        BenchmarkSample(
                            prompt_index=prompt_index,
                            run_index=run_index,
                            latency_seconds=0,
                            error=str(error),
                        )
                    )
        return BenchmarkResult(
            model=request.model,
            backend=request.backend,
            resolved_checkpoint=checkpoint,
            profile=request.profile,
            samples=samples,
            mean_latency_seconds=mean_latency(samples),
            mean_tokens_per_second=mean_tokens_per_second(samples),
        )

    @staticmethod
    def _chat_request(request: BenchmarkRequest, prompt: str) -> ChatRequest:
        return ChatRequest(
            model=request.model,
            backend=request.backend,
            prompt=prompt,
            generation=request.generation,
        )
