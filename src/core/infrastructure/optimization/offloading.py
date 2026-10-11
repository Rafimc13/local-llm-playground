"""CPU/GPU offloading constraints."""


def validate_cpu_offload_gb(cpu_offload_gb: float) -> None:
    """Validate a bounded amount of host-memory offloading."""
    maximum_offload_gb = 64
    if not 0 <= cpu_offload_gb <= maximum_offload_gb:
        msg = "cpu_offload_gb must be between 0 and 64"
        raise ValueError(msg)
