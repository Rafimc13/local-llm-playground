"""Backend interface shared by all inference engines."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.core.schemas.chat import ChatRequest, ChatResponse
from src.core.schemas.model import BackendInfo, BackendName


class BackendError(RuntimeError):
    """Raised when an inference engine cannot satisfy a request."""


class LLMBackend(ABC):
    """Backend-neutral asynchronous inference interface."""

    name: BackendName

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        """Initialize a backend endpoint and request timeout."""
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def health(self) -> BackendInfo:
        """Report whether the external inference server is reachable."""

    @abstractmethod
    async def chat(self, request: ChatRequest, checkpoint: str) -> ChatResponse:
        """Generate a complete response."""

    @abstractmethod
    def stream(self, request: ChatRequest, checkpoint: str) -> AsyncIterator[str]:
        """Yield generated text fragments."""
