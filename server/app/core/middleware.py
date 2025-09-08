import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import client_ip, endpoint, get_logger, request_id

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add API-appropriate security headers to all responses."""

    def __init__(self, app: Any, headers: dict[str, str] | None = None) -> None:
        """Init middleware with minimal headers for API behind AWS API Gateway + WAF."""
        super().__init__(app)
        self.security_headers = headers or {
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers to response
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging with timing and context."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Log request details, execute handler, and log response w/ timing metrics."""
        start_time = time.time()
        req_id = str(uuid.uuid4())

        request_id.set(req_id)
        client_ip.set(request.client.host if request.client else "")
        endpoint.set(f"{request.method} {request.url.path}")

        # Use debug level for healthcheck requests to avoid log pollution
        log_level = logger.debug if request.url.path == "/v1/health" else logger.info
        log_level(
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

            # Use debug level for healthcheck requests to avoid log pollution
            log_level = (
                logger.debug if request.url.path == "/v1/health" else logger.info
            )
            log_level(
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
