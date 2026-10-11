"""Shared adapter for vLLM and SGLang OpenAI-compatible APIs."""

from collections.abc import AsyncIterator
from time import perf_counter

import httpx
from pydantic import ValidationError

from src.core.infrastructure.backends.base import BackendError, LLMBackend
from src.core.infrastructure.backends.wire import OpenAIChatResponse, OpenAIStreamEvent
from src.core.schemas.chat import ChatRequest, ChatResponse
from src.core.schemas.model import BackendInfo


class OpenAICompatibleBackend(LLMBackend):
    """Common implementation for local OpenAI-compatible servers."""

    supports_offloading = True
    supports_continuous_batching = True

    async def health(self) -> BackendInfo:
        """Check the model-list endpoint without running inference."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
                _ = response.raise_for_status()
        except httpx.HTTPError as error:
            return BackendInfo(
                name=self.name,
                base_url=self.base_url,
                healthy=False,
                detail=str(error),
                supports_offloading=self.supports_offloading,
                supports_continuous_batching=self.supports_continuous_batching,
            )
        return BackendInfo(
            name=self.name,
            base_url=self.base_url,
            healthy=True,
            supports_offloading=self.supports_offloading,
            supports_continuous_batching=self.supports_continuous_batching,
        )

    async def chat(self, request: ChatRequest, checkpoint: str) -> ChatResponse:
        """Run a non-streaming OpenAI-compatible chat request."""
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=self.build_payload(request, checkpoint, stream=False),
                )
                _ = response.raise_for_status()
            parsed = OpenAIChatResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError) as error:
            msg = f"{self.name.value} request failed: {error}"
            raise BackendError(msg) from error
        latency = perf_counter() - started
        completion_tokens = parsed.usage.completion_tokens if parsed.usage else None
        tokens_per_second = completion_tokens / latency if completion_tokens else None
        return ChatResponse(
            model=request.model,
            backend=self.name,
            resolved_checkpoint=checkpoint,
            content=parsed.choices[0].message.content or "",
            prompt_tokens=parsed.usage.prompt_tokens if parsed.usage else None,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            tokens_per_second=tokens_per_second,
        )

    async def stream(self, request: ChatRequest, checkpoint: str) -> AsyncIterator[str]:
        """Yield text fragments from an OpenAI-compatible SSE stream."""
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout_seconds) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=self.build_payload(request, checkpoint, stream=True),
                ) as response,
            ):
                _ = response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    parsed = OpenAIStreamEvent.model_validate_json(line[6:])
                    if parsed.choices and parsed.choices[0].delta.content:
                        yield parsed.choices[0].delta.content or ""
        except (httpx.HTTPError, ValidationError) as error:
            msg = f"{self.name.value} stream failed: {error}"
            raise BackendError(msg) from error

    @staticmethod
    def build_payload(request: ChatRequest, checkpoint: str, *, stream: bool) -> dict[str, object]:
        """Translate a neutral chat request into OpenAI's wire format."""
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        generation = request.generation
        return {
            "model": checkpoint,
            "messages": messages,
            "stream": stream,
            "max_tokens": generation.max_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "seed": generation.seed,
            "chat_template_kwargs": {"enable_thinking": generation.thinking},
        }
