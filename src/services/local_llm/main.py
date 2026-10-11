"""FastAPI application factory and development entry point."""

import uvicorn
from fastapi import FastAPI

from src.services.local_llm.api.router import api_router


def _health() -> dict[str, str]:
    """Report API-process health without probing inference engines."""
    return {"status": "ok"}


def create_app() -> FastAPI:
    """Create the Local LLM Playground API."""
    application = FastAPI(
        title="Local LLM Playground",
        version="0.1.0",
        description="Unified local inference and benchmarking API.",
    )
    application.include_router(api_router)

    application.add_api_route("/health", _health, methods=["GET"], tags=["system"])

    return application


app = create_app()


def main() -> None:
    """Run the development API server."""
    uvicorn.run("src.services.local_llm.main:app", host="127.0.0.1", port=8080, reload=True)


if __name__ == "__main__":
    main()
