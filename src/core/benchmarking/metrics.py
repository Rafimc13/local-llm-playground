"""Benchmark aggregation helpers."""

from statistics import fmean

from src.core.schemas.benchmark import BenchmarkSample


def mean_latency(samples: list[BenchmarkSample]) -> float | None:
    """Calculate mean latency across successful samples."""
    values = [sample.latency_seconds for sample in samples if sample.error is None]
    return fmean(values) if values else None


def mean_tokens_per_second(samples: list[BenchmarkSample]) -> float | None:
    """Calculate mean decode throughput when token counts are available."""
    values = [
        sample.tokens_per_second
        for sample in samples
        if sample.error is None and sample.tokens_per_second is not None
    ]
    return fmean(values) if values else None
