"""Top-level service router."""

from fastapi import APIRouter

from src.core.config.constants import API_PREFIX
from src.services.local_llm.api.endpoints import router as local_llm_router

api_router = APIRouter()
api_router.include_router(local_llm_router, prefix=API_PREFIX, tags=["local-llm"])
