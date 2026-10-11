"""Persist benchmark results in machine-readable formats."""

import csv
from pathlib import Path

from src.core.schemas.benchmark import BenchmarkResult


def write_json(result: BenchmarkResult, path: Path) -> None:
    """Write a complete benchmark record as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def write_csv(result: BenchmarkResult, path: Path) -> None:
    """Write individual benchmark samples as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        header = (
            "model",
            "backend",
            "checkpoint",
            "profile",
            "prompt_index",
            "run_index",
            "latency_seconds",
            "completion_tokens",
            "tokens_per_second",
            "peak_vram_mib",
            "error",
        )
        writer = csv.writer(handle)
        _ = writer.writerow(header)
        for sample in result.samples:
            _ = writer.writerow(
                (
                    result.model,
                    result.backend.value,
                    result.resolved_checkpoint,
                    result.profile,
                    sample.prompt_index,
                    sample.run_index,
                    sample.latency_seconds,
                    sample.completion_tokens,
                    sample.tokens_per_second,
                    sample.peak_vram_mib,
                    sample.error,
                )
            )
