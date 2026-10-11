"""Benchmark configuration and result schemas."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.core.schemas.chat import GenerationOptions
from src.core.schemas.model import BackendName


class BenchmarkRequest(BaseModel):
    """Request for a repeatable prompt benchmark."""

    model: str
    backend: BackendName
    prompts: list[str] = Field(min_length=1, max_length=100)
    profile: str = "low-memory"
    warmup_runs: int = Field(default=1, ge=0, le=10)
    measured_runs: int = Field(default=3, ge=1, le=100)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class BenchmarkSample(BaseModel):
    """Metrics collected for one inference request."""

    prompt_index: int
    run_index: int
    latency_seconds: float
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    peak_vram_mib: float | None = None
    error: str | None = None


class BenchmarkResult(BaseModel):
    """Complete benchmark output with reproducibility metadata."""

    model: str
    backend: BackendName
    resolved_checkpoint: str
    profile: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    samples: list[BenchmarkSample]
    mean_latency_seconds: float | None = None
    mean_tokens_per_second: float | None = None
