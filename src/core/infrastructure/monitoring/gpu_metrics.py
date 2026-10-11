"""GPU telemetry collected through the stable nvidia-smi CLI."""

import csv
import shutil
import subprocess
from dataclasses import dataclass
from io import StringIO


@dataclass(frozen=True, slots=True)
class GPUMetrics:
    """One NVIDIA GPU utilization snapshot."""

    index: int
    name: str
    driver_version: str
    memory_total_mib: float
    memory_used_mib: float
    memory_free_mib: float
    utilization_percent: float
    temperature_celsius: float

    def to_dict(self) -> dict[str, int | str | float]:
        """Return a JSON-serializable representation."""
        return {
            "index": self.index,
            "name": self.name,
            "driver_version": self.driver_version,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_free_mib": self.memory_free_mib,
            "utilization_percent": self.utilization_percent,
            "temperature_celsius": self.temperature_celsius,
        }


class GPUUnavailableError(RuntimeError):
    """Raised when NVIDIA telemetry cannot be collected."""


def read_gpu_metrics() -> tuple[GPUMetrics, ...]:
    """Read all visible NVIDIA devices without initializing a model runtime."""
    expected_columns = 8
    query = (
        "index,name,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu"
    )
    executable = shutil.which("nvidia-smi")
    if executable is None:
        msg = "NVIDIA GPU metrics unavailable: nvidia-smi was not found"
        raise GPUUnavailableError(msg)
    try:
        result = subprocess.run(  # noqa: S603
            (executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        msg = f"NVIDIA GPU metrics unavailable: {error}"
        raise GPUUnavailableError(msg) from error

    metrics: list[GPUMetrics] = []
    reader = csv.reader(StringIO(result.stdout))
    for row in reader:
        values = [value.strip() for value in row]
        if len(values) != expected_columns:
            msg = f"Unexpected nvidia-smi output row: {row}"
            raise GPUUnavailableError(msg)
        metrics.append(
            GPUMetrics(
                index=int(values[0]),
                name=values[1],
                driver_version=values[2],
                memory_total_mib=float(values[3]),
                memory_used_mib=float(values[4]),
                memory_free_mib=float(values[5]),
                utilization_percent=float(values[6]),
                temperature_celsius=float(values[7]),
            )
        )
    return tuple(metrics)
