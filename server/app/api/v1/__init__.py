from fastapi import APIRouter

from .endpoints import router as _endpoints_router
from .feedback import feedback_router

# Combine all v1 routers so main.py requires no changes.
router = APIRouter()
router.include_router(_endpoints_router)
router.include_router(feedback_router)

__all__ = ["router"]
