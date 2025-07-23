# FILE: ./app/core/task_processors.py

from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.storage import StorageBackend
from app.core.types import ProjectStatus, TaskType
from app.db.database import SessionLocal
from app.services.inference_service import InferenceService
from app.services.project_service import ProjectService

logger = get_logger(__name__)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions in background tasks."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def process_task(
    storage: StorageBackend, task_type: str, task_data: dict[str, Any]
) -> dict[str, Any]:
    """Generic task processor that handles common database and project management."""
    project_id = task_data["project_id"]

    with get_db_session() as db:
        project_service = ProjectService(storage, db)
        inference_service = InferenceService(storage, db, project_service)

        project_service.update_project_status(project_id, ProjectStatus.RUNNING)

        if task_type == TaskType.INFERENCE.value:
            result = await _handle_inference(task_data, inference_service)
        elif task_type == TaskType.POLYGONIZE.value:
            result = await _handle_polygonize(task_data, inference_service)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        if result.get("inference_file") or result.get("polygon_file"):
            task_type_enum = TaskType(task_type)
            project_service.record_task_completion(project_id, task_type_enum, result)

        return result


async def _handle_inference(
    task_data: dict, inference_service: InferenceService
) -> dict:
    """Handle inference-specific processing."""
    project_id = task_data["project_id"]
    params = task_data["inference_params"]
    return await inference_service.run_project_inference(project_id, params)


async def _handle_polygonize(
    task_data: dict, inference_service: InferenceService
) -> dict:
    """Handle polygonization-specific processing."""
    project_id = task_data["project_id"]
    params = task_data["polygon_params"]
    return await inference_service.run_project_polygonize(project_id, params)


def get_task_processors(storage: StorageBackend) -> dict[str, Any]:
    """Get dictionary of task type to processor function mappings."""
    base_processor = partial(process_task, storage)

    return {
        TaskType.INFERENCE.value: partial(base_processor, TaskType.INFERENCE.value),
        TaskType.POLYGONIZE.value: partial(base_processor, TaskType.POLYGONIZE.value),
    }
