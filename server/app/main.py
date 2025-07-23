import contextlib
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import router as v1_router
from app.core.config import get_settings
from app.core.logging import AppLogger, get_logger
from app.core.middleware import LoggingMiddleware, SecurityHeadersMiddleware
from app.core.queue import InMemoryQueue, QueueBackend, get_queue
from app.core.storage import StorageBackend, get_storage
from app.core.task_processors import get_task_processors
from app.db.database import create_db_and_tables

logger = get_logger(__name__)


# Exception handlers
def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI validation errors with detailed logging."""
    first_error = (
        exc.errors()[0] if exc.errors() else {"msg": "Unknown validation error"}
    )
    logger.warning(
        "Validation error",
        extra={
            "error_details": exc.errors(),
            "request_url": str(request.url),
            "request_method": request.method,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": first_error["msg"]},
    )


# Application initialization functions
async def initialize_logging() -> None:
    """Initialize application logging."""
    AppLogger()
    logger.info("Logging initialized")


async def initialize_database() -> None:
    """Initialize database and create tables."""
    try:
        create_db_and_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def initialize_services() -> tuple[StorageBackend, QueueBackend]:
    """Initialize storage, task processors, and queue services."""
    try:
        settings = get_settings()

        storage = get_storage(settings)
        task_processors = get_task_processors(storage)
        queue = get_queue(settings, task_processors)

        logger.info("Services initialized")
        return storage, queue
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


async def start_background_workers(queue: QueueBackend) -> None:
    """Start background queue workers if needed."""
    if isinstance(queue, InMemoryQueue):
        await queue.start_workers()
        logger.info("Queue workers started")


async def stop_background_workers(queue: QueueBackend) -> None:
    """Stop background queue workers if needed."""
    try:
        if isinstance(queue, InMemoryQueue):
            await queue.stop_workers()
            logger.info("Queue workers stopped")
    except Exception as e:
        logger.error(f"Error stopping background workers: {e}")
        # Don't re-raise during shutdown to avoid blocking app shutdown


def setup_app_state(app: FastAPI, storage: StorageBackend, queue: QueueBackend) -> None:
    """Setup application state with initialized services."""
    app.state.queue = queue  # type: ignore[attr-defined]
    app.state.storage = storage  # type: ignore[attr-defined]
    logger.info("Application state configured")


# Application lifespan management
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Application starting up")

    await initialize_logging()
    await initialize_database()

    storage, queue = await initialize_services()
    setup_app_state(app, storage, queue)

    await start_background_workers(queue)

    logger.info("Application startup complete")

    yield

    logger.info("Application shutting down")

    await stop_background_workers(app.state.queue)  # type: ignore[attr-defined]

    logger.info("Application shutdown complete")


# FastAPI application factory
def create_app() -> FastAPI:
    """Create FastAPI application with lifespan management."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        lifespan=lifespan,
    )

    # Security headers first
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging middleware last
    app.add_middleware(LoggingMiddleware)

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,  # type: ignore[arg-type]
    )

    app.include_router(v1_router, prefix="/v1")

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=get_settings().host,
        port=get_settings().port,
        reload=get_settings().debug,
    )
