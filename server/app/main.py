import contextlib

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.endpoints import router as api_router
from app.core.config import get_settings
from app.core.logging import AppLogger, get_logger
from app.core.middleware import LoggingMiddleware
from app.db.database import create_db_and_tables

logger = get_logger(__name__)


# Define lifespan context manager
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize logging and create database tables
    AppLogger()
    logger.info("Application starting up")
    create_db_and_tables()
    yield
    # Shutdown: cleanup would go here (if needed)
    logger.info("Application shutting down")


# Create FastAPI app with lifespan
app = FastAPI(
    title=get_settings().api_title,
    description=get_settings().api_description,
    version=get_settings().api_version,
    lifespan=lifespan,
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
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


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=get_settings().host,
        port=get_settings().port,
        reload=get_settings().debug,
    )
