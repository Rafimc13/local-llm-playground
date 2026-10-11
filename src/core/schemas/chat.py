"""Chat completion request and response schemas."""

from pydantic import BaseModel, Field

from src.core.config.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from src.core.schemas.model import BackendName


class GenerationOptions(BaseModel):
    """Backend-neutral text generation controls."""

    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=8192)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    seed: int | None = None
    thinking: bool = False


class ChatRequest(BaseModel):
    """Unified chat request accepted by every backend."""

    model: str
    backend: BackendName
    prompt: str = Field(min_length=1, max_length=100_000)
    system_prompt: str | None = Field(default=None, max_length=20_000)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class ChatResponse(BaseModel):
    """Normalized non-streaming completion response."""

    model: str
    backend: BackendName
    resolved_checkpoint: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float
    tokens_per_second: float | None = None
