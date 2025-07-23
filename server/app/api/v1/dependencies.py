from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.auth import verify_auth
from app.core.queue import QueueBackend
from app.core.storage import StorageBackend
from app.db.database import get_db
from app.services.inference_service import InferenceService
from app.services.project_service import ProjectService
from app.services.task_service import TaskService


def get_queue_service(request: Request) -> QueueBackend:
    """Get queue service from app state"""
    return request.app.state.queue


def get_storage_service(request: Request) -> StorageBackend:
    """Get storage service from app state"""
    return request.app.state.storage


def get_task_service(queue: QueueBackend = Depends(get_queue_service)) -> TaskService:
    return TaskService(queue)


def get_project_service(
    storage: StorageBackend = Depends(get_storage_service),
    db: Session = Depends(get_db),
) -> ProjectService:
    return ProjectService(storage, db)


def get_inference_service_with_storage(
    storage: StorageBackend = Depends(get_storage_service),
    db: Session = Depends(get_db),
    project_service: ProjectService = Depends(get_project_service),
    task_service: TaskService = Depends(get_task_service),
) -> InferenceService:
    return InferenceService(storage, db, project_service, task_service)


# Type aliases for easier dependency injection
AuthDep = Annotated[dict, Depends(verify_auth)]
DBDep = Annotated[Session, Depends(get_db)]
QueueDep = Annotated[QueueBackend, Depends(get_queue_service)]
StorageDep = Annotated[StorageBackend, Depends(get_storage_service)]
InferenceServiceDep = Annotated[
    InferenceService, Depends(get_inference_service_with_storage)
]
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
