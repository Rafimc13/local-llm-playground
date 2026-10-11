"""In-process inference timing utilities."""

from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class InferenceTimer:
    """Small monotonic timer used by benchmark integrations."""

    started_at: float = 0.0

    def start(self) -> None:
        """Start or reset the timer."""
        self.started_at = perf_counter()

    def elapsed(self) -> float:
        """Return seconds elapsed since the latest start."""
        if self.started_at == 0:
            msg = "Timer has not been started"
            raise RuntimeError(msg)
        return perf_counter() - self.started_at
