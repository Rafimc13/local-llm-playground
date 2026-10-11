"""Application service tests with a fully in-memory backend."""

from collections.abc import AsyncIterator

from src.core.infrastructure.backends.base import LLMBackend
from src.core.local_llm_models.registry import model_registry
from src.core.schemas.chat import ChatRequest, ChatResponse
from src.core.schemas.model import BackendInfo, BackendName
from src.services.local_llm.service import LocalLLMService


class FakeBackend(LLMBackend):
    """Deterministic backend used for service tests."""

    name = BackendName.OLLAMA

    async def health(self) -> BackendInfo:
        return BackendInfo(name=self.name, base_url=self.base_url, healthy=True)

    async def chat(self, request: ChatRequest, checkpoint: str) -> ChatResponse:
        return ChatResponse(
            model=request.model,
            backend=request.backend,
            resolved_checkpoint=checkpoint,
            content="test response",
            latency_seconds=0.01,
        )

    async def stream(self, request: ChatRequest, checkpoint: str) -> AsyncIterator[str]:
        del request, checkpoint
        yield "test"


class FakeFactory:
    """Factory returning an in-memory backend."""

    def create(self, name: BackendName) -> LLMBackend:
        backend = FakeBackend("http://test", 1)
        backend.name = name
        return backend


async def test_service_resolves_checkpoint_before_chat() -> None:
    service = LocalLLMService(model_registry, FakeFactory())
    response = await service.chat(
        ChatRequest(model="qwen3-1.7b", backend=BackendName.OLLAMA, prompt="Hello")
    )
    assert response.resolved_checkpoint == "qwen3:1.7b-q4_K_M"
    assert response.content == "test response"
