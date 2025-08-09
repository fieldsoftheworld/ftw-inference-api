from typing import Annotated, cast

from fastapi import Depends, Request

from app.core.auth import verify_auth
from app.core.queue import QueueBackend
from app.core.storage import StorageBackend
from app.services.inference_service import InferenceService
from app.services.project_service import ProjectService
from app.services.task_service import TaskService


def get_queue_service(request: Request) -> QueueBackend:
    """Get queue backend service from app state for task processing."""
    return cast("QueueBackend", request.app.state.queue)


def get_storage_service(request: Request) -> StorageBackend:
    """Get storage backend service from app state for file operations."""
    return cast("StorageBackend", request.app.state.storage)


def get_task_service(queue: QueueBackend = Depends(get_queue_service)) -> TaskService:
    """Create TaskService instance with queue backend for managing async tasks."""
    return TaskService(queue)


def get_project_service(
    storage: StorageBackend = Depends(get_storage_service),
) -> ProjectService:
    """Create ProjectService instance with storage backend for project operations."""
    return ProjectService(storage)


def get_inference_service_with_storage(
    storage: StorageBackend = Depends(get_storage_service),
    project_service: ProjectService = Depends(get_project_service),
    task_service: TaskService = Depends(get_task_service),
) -> InferenceService:
    """Create InferenceService instance with dependencies for ML inference."""
    return InferenceService(storage, project_service, task_service)


# Type aliases for easier dependency injection
AuthDep = Annotated[dict, Depends(verify_auth)]
QueueDep = Annotated[QueueBackend, Depends(get_queue_service)]
StorageDep = Annotated[StorageBackend, Depends(get_storage_service)]
InferenceServiceDep = Annotated[
    InferenceService, Depends(get_inference_service_with_storage)
]
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
