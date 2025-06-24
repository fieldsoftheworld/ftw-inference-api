import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import client_ip, endpoint, get_logger, request_id

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        req_id = str(uuid.uuid4())

        request_id.set(req_id)
        client_ip.set(request.client.host if request.client else "")
        endpoint.set(f"{request.method} {request.url.path}")

        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent", ""),
            },
        )

        try:
            response = await call_next(request)

            duration_ms = round((time.time() - start_time) * 1000, 2)

            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )

            return response

        except Exception as exc:
            duration_ms = round((time.time() - start_time) * 1000, 2)

            logger.error(
                "Request failed",
                exc_info=True,
                extra={
                    "method": request.method,
                    "duration_ms": duration_ms,
                    "error_type": type(exc).__name__,
                },
            )

            raise exc
