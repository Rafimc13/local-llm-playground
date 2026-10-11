"""Model registry tests."""

import pytest

from src.core.local_llm_models.registry import model_registry
from src.core.schemas.model import BackendName


def test_registry_contains_expected_models() -> None:
    assert [model.id for model in model_registry.list()] == [
        "qwen3-0.6b",
        "qwen3-1.7b",
        "qwen3-4b",
    ]


def test_registry_resolves_backend_specific_checkpoint() -> None:
    checkpoint = model_registry.resolve_checkpoint("qwen3-4b", BackendName.VLLM)
    assert checkpoint == "Qwen/Qwen3-4B-AWQ"


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        _ = model_registry.get("missing")
