"""Performance suite guard."""

import pytest


@pytest.mark.benchmark
def test_benchmarks_are_never_part_of_cpu_ci() -> None:
    pytest.skip("Run benchmarks manually against a selected local engine.")
