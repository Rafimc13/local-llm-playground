"""Simple memory planning helpers."""


def available_vram_budget_mib(total_mib: float, utilization: float) -> float:
    """Calculate the configured engine memory budget."""
    if total_mib <= 0:
        msg = "total_mib must be positive"
        raise ValueError(msg)
    if not 0 < utilization <= 1:
        msg = "utilization must be greater than 0 and at most 1"
        raise ValueError(msg)
    return total_mib * utilization
