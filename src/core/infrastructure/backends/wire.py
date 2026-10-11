"""Typed wire formats used to validate external backend responses."""

from pydantic import BaseModel, Field


class OllamaMessage(BaseModel):
    """Message in the Ollama chat protocol."""

    role: str
    content: str


class OllamaChatResponse(BaseModel):
    """Relevant fields from an Ollama chat response."""

    message: OllamaMessage
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OpenAIMessage(BaseModel):
    """Assistant message from an OpenAI-compatible endpoint."""

    content: str | None = None


class OpenAIChoice(BaseModel):
    """Completion choice from an OpenAI-compatible endpoint."""

    message: OpenAIMessage


class OpenAIUsage(BaseModel):
    """Token accounting from an OpenAI-compatible endpoint."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class OpenAIChatResponse(BaseModel):
    """Relevant fields from an OpenAI-compatible chat response."""

    choices: list[OpenAIChoice] = Field(min_length=1)
    usage: OpenAIUsage | None = None


class StreamDelta(BaseModel):
    """Text delta in an OpenAI-compatible stream event."""

    content: str | None = None


class StreamChoice(BaseModel):
    """Streaming completion choice."""

    delta: StreamDelta


class OpenAIStreamEvent(BaseModel):
    """One OpenAI-compatible server-sent event payload."""

    choices: list[StreamChoice] = Field(default_factory=list)
