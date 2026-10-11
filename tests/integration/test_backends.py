"""Placeholder for explicit, opt-in backend integration runs."""

import pytest


@pytest.mark.integration
def test_integration_suite_is_opt_in() -> None:
    pytest.skip("Start exactly one backend, then add its explicit integration selection.")
