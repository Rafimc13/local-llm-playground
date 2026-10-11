"""Native Ollama HTTP adapter."""

from collections.abc import AsyncIterator
from time import perf_counter

import httpx
from pydantic import ValidationError

from src.core.infrastructure.backends.base import BackendError, LLMBackend
from src.core.infrastructure.backends.wire import OllamaChatResponse
from src.core.schemas.chat import ChatRequest, ChatResponse
from src.core.schemas.model import BackendInfo, BackendName


class OllamaBackend(LLMBackend):
    """Connect to Ollama's native REST API."""

    name = BackendName.OLLAMA

    async def health(self) -> BackendInfo:
        """Check the Ollama tags endpoint without loading a model."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                _ = response.raise_for_status()
        except httpx.HTTPError as error:
            return BackendInfo(
                name=self.name,
                base_url=self.base_url,
                healthy=False,
                detail=str(error),
                supports_offloading=True,
            )
        return BackendInfo(
            name=self.name,
            base_url=self.base_url,
            healthy=True,
            supports_offloading=True,
        )

    async def chat(self, request: ChatRequest, checkpoint: str) -> ChatResponse:
        """Run a non-streaming Ollama chat request."""
        started = perf_counter()
        payload = self.build_payload(request, checkpoint, stream=False)
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                _ = response.raise_for_status()
            parsed = OllamaChatResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError) as error:
            msg = f"Ollama request failed: {error}"
            raise BackendError(msg) from error
        latency = perf_counter() - started
        tokens_per_second = None
        if parsed.eval_count and parsed.eval_duration:
            tokens_per_second = parsed.eval_count / (parsed.eval_duration / 1_000_000_000)
        return ChatResponse(
            model=request.model,
            backend=self.name,
            resolved_checkpoint=checkpoint,
            content=parsed.message.content,
            prompt_tokens=parsed.prompt_eval_count,
            completion_tokens=parsed.eval_count,
            latency_seconds=latency,
            tokens_per_second=tokens_per_second,
        )

    async def stream(self, request: ChatRequest, checkpoint: str) -> AsyncIterator[str]:
        """Yield text fragments from Ollama's newline-delimited JSON response."""
        payload = self.build_payload(request, checkpoint, stream=True)
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout_seconds) as client,
                client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response,
            ):
                _ = response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    parsed = OllamaChatResponse.model_validate_json(line)
                    if parsed.message.content:
                        yield parsed.message.content
        except (httpx.HTTPError, ValidationError) as error:
            msg = f"Ollama stream failed: {error}"
            raise BackendError(msg) from error

    @staticmethod
    def build_payload(request: ChatRequest, checkpoint: str, *, stream: bool) -> dict[str, object]:
        """Translate a neutral chat request into Ollama's wire format."""
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        generation = request.generation
        return {
            "model": checkpoint,
            "messages": messages,
            "stream": stream,
            "think": generation.thinking,
            "options": {
                "num_predict": generation.max_tokens,
                "temperature": generation.temperature,
                "top_p": generation.top_p,
                "seed": generation.seed,
            },
        }
