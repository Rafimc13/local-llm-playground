"""FastAPI smoke tests."""

from fastapi.testclient import TestClient

from src.services.local_llm.main import create_app


def test_health() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_does_not_require_a_running_engine() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/local-llm/models")
    assert response.status_code == 200
    assert len(response.json()) == 3
