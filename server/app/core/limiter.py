import asyncio
from functools import wraps

import pendulum
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.example import ExampleRequestLog


def setup_logs(db) -> ExampleRequestLog | None:
    settings = get_settings()

    # Step 1: Cleanup old entries
    threshold = pendulum.now("UTC").subtract(seconds=settings.example_timeout + 1)
    db.query(ExampleRequestLog).filter(
        ExampleRequestLog.created_at < threshold
    ).delete()
    db.commit()

    # Step 2: Count active requests
    active_count = db.query(ExampleRequestLog).count()
    if active_count >= settings.max_concurrent_examples:
        return None

    # Step 3: Log this request
    log = ExampleRequestLog()
    db.add(log)
    db.commit()
    db.refresh(log)

    return log


def delete_logs(db: Session, log: ExampleRequestLog | None):
    if log is not None:
        db.delete(log)
        db.commit()


def timeout():
    settings = get_settings()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=settings.example_timeout
                )
            except asyncio.TimeoutError as e:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request timed out. Please try a smaller area.",
                ) from e

        return wrapper

    return decorator
