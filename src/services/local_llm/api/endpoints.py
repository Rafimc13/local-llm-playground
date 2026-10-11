"""Local LLM REST endpoints."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.core.infrastructure.backends.base import BackendError
from src.core.infrastructure.monitoring.gpu_metrics import GPUUnavailableError
from src.core.schemas.benchmark import BenchmarkRequest, BenchmarkResult
from src.core.schemas.chat import ChatRequest, ChatResponse
from src.core.schemas.model import BackendInfo, ModelInfo
from src.services.local_llm.dependencies import get_service
from src.services.local_llm.service import LocalLLMService

ServiceDependency = Annotated[LocalLLMService, Depends(get_service)]

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
def list_models(service: ServiceDependency) -> tuple[ModelInfo, ...]:
    """List models and exact backend checkpoint mappings."""
    return service.list_models()


@router.get("/backends", response_model=list[BackendInfo])
async def list_backends(service: ServiceDependency) -> tuple[BackendInfo, ...]:
    """Report backend reachability and capabilities."""
    return await service.list_backends()


@router.post("/chat")
async def chat(request: ChatRequest, service: ServiceDependency) -> ChatResponse:
    """Generate a complete chat response."""
    try:
        return await service.chat(request)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except BackendError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest, service: ServiceDependency) -> StreamingResponse:
    """Stream text fragments as server-sent events."""
    iterator = service.stream(request)

    async def events() -> AsyncIterator[str]:
        try:
            async for fragment in iterator:
                yield f"data: {json.dumps({'content': fragment})}\n\n"
            yield "data: [DONE]\n\n"
        except (BackendError, ValueError) as error:
            yield f"event: error\ndata: {json.dumps({'detail': str(error)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/benchmarks")
async def benchmark(request: BenchmarkRequest, service: ServiceDependency) -> BenchmarkResult:
    """Execute a benchmark against an already-running backend."""
    try:
        return await service.benchmark(request)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    except BackendError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


@router.get("/system/gpu")
def gpu_metrics(service: ServiceDependency) -> list[dict[str, int | str | float]]:
    """Return a current nvidia-smi snapshot."""
    try:
        return [metric.to_dict() for metric in service.gpu_metrics()]
    except GPUUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
