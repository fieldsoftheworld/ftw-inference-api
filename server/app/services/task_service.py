from typing import Any

from fastapi import HTTPException, status

from app.core.queue import QueueBackend
from app.core.types import TaskStatus, TaskType


class TaskService:
    def __init__(self, queue: QueueBackend):
        self.queue = queue

    # --- Public API: Task Submission ---

    async def submit_inference_task(
        self, project_id: str, inference_params: dict[str, Any]
    ) -> str:
        """Submit an inference task to the queue."""
        payload = {
            "project_id": project_id,
            "inference_params": inference_params,
        }
        return await self.queue.submit(TaskType.INFERENCE, payload)

    async def submit_polygonize_task(
        self, project_id: str, polygon_params: dict[str, Any]
    ) -> str:
        """Submit a polygonization task to the queue."""
        payload = {
            "project_id": project_id,
            "polygon_params": polygon_params,
        }
        return await self.queue.submit(TaskType.POLYGONIZE, payload)

    # --- Public API: Task Retrieval ---

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get the raw status enum for a given task."""
        task_info = await self.queue.get_status(task_id)
        return task_info.status

    async def get_task_info(self, task_id: str) -> dict[str, Any] | None:
        """Get formatted task information for embedding in a project status response."""
        try:
            task_info = await self.queue.get_status(task_id)
            return self._format_task_info(task_info)
        except ValueError:
            return None

    async def get_task_details(self, project_id: str, task_id: str) -> dict[str, Any]:
        """Get detailed, formatted task information for a specific task endpoint."""
        try:
            task_info = await self.queue.get_status(task_id)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found",
            ) from e

        return self._format_task_details(
            task_info, task_id=task_id, project_id=project_id
        )

    # --- Internal Helper Methods ---

    def _format_task_info(
        self, task_info, task_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        """Format task info for TaskInfoResponse (project status)."""
        return {
            "task_id": task_id or task_info.task_id,
            "project_id": project_id,
            "task_type": task_info.task_type,
            "task_status": task_info.status.value,
            "created_at": task_info.created_at.isoformat(),
            "started_at": task_info.started_at.isoformat()
            if task_info.started_at
            else None,
            "completed_at": task_info.completed_at.isoformat()
            if task_info.completed_at
            else None,
            "error": task_info.error,
            "result": task_info.result,
        }

    def _format_task_details(
        self, task_info, task_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        """Format task info for TaskDetailsResponse."""
        return {
            "task_id": task_id or task_info.task_id,
            "project_id": project_id,
            "task_type": task_info.task_type,
            "status": task_info.status.value,
            "created_at": task_info.created_at.isoformat(),
            "started_at": task_info.started_at.isoformat()
            if task_info.started_at
            else None,
            "completed_at": task_info.completed_at.isoformat()
            if task_info.completed_at
            else None,
            "error": task_info.error,
            "result": task_info.result,
        }
