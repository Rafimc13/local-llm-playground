"""Batching constraints."""


def validate_concurrency(concurrency: int, *, maximum: int = 4) -> None:
    """Keep consumer-GPU experiments inside the configured envelope."""
    if not 1 <= concurrency <= maximum:
        message = f"concurrency must be between 1 and {maximum}"
        raise ValueError(message)
