import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import pendulum

from app.core.logging import get_logger
from app.core.types import TaskStatus, TaskType

logger = get_logger(__name__)


@dataclass
class TaskInfo:
    """Task information returned by queue"""

    task_id: str
    status: TaskStatus
    task_type: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None


class QueueBackend(Protocol):
    """Protocol for task queue operations"""

    async def submit(self, task_type: TaskType, payload: dict) -> str:
        """Submit task and return task ID"""
        ...

    async def get_status(self, task_id: str) -> TaskInfo:
        """Get task status and info"""
        ...

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending task"""
        ...


class InMemoryQueue:
    """Current in-memory implementation - will be replaced by SQS"""

    def __init__(
        self,
        max_workers: int = 2,
        task_processors: dict[str, Callable] | None = None,
    ) -> None:
        """Initialize in-memory queue with worker pool."""
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self.active_tasks: dict[str, dict[str, Any]] = {}
        self.max_workers = max_workers
        self.shutdown_event = asyncio.Event()
        self.task_processors = task_processors or {}

    async def submit(self, task_type: TaskType, payload: dict) -> str:
        """Submit task to queue and return task ID."""
        task_id = str(uuid.uuid4())
        task_type_str = task_type.value
        task_data = {"id": task_id, "task_type": task_type_str, **payload}

        created_at = pendulum.now("UTC").isoformat()
        self.active_tasks[task_id] = {
            "status": TaskStatus.PENDING.value,
            "task_type": task_type_str,
            "project_id": payload.get("project_id"),
            "created_at": created_at,
            "updated_at": created_at,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
        }

        await self.queue.put(task_data)
        logger.info(f"Submitted {task_type_str} task {task_id}")
        return task_id

    async def get_status(self, task_id: str) -> TaskInfo:
        """Get task status and info."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")

        task_data = self.active_tasks[task_id]
        started_at = None
        if task_data["started_at"]:
            started_at = datetime.fromisoformat(task_data["started_at"])

        completed_at = None
        if task_data["completed_at"]:
            completed_at = datetime.fromisoformat(task_data["completed_at"])

        return TaskInfo(
            task_id=task_id,
            status=TaskStatus(task_data["status"]),
            task_type=task_data["task_type"],
            created_at=datetime.fromisoformat(task_data["created_at"]),
            updated_at=datetime.fromisoformat(
                task_data.get("updated_at", task_data["created_at"])
            ),
            started_at=started_at,
            completed_at=completed_at,
            result=task_data.get("result"),
            error=task_data.get("error"),
        )

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        if task_id not in self.active_tasks:
            return False

        task_info = self.active_tasks[task_id]
        if task_info["status"] == TaskStatus.PENDING.value:
            now = pendulum.now("UTC").isoformat()
            task_info["status"] = TaskStatus.FAILED.value
            task_info["error"] = "Task cancelled"
            task_info["updated_at"] = now
            task_info["completed_at"] = now
            logger.info(f"Cancelled task {task_id}")
            return True

        return False

    async def start_workers(self) -> None:
        """Start background workers."""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self.worker(f"worker-{i}"))
            self.workers.append(worker)
        logger.info(f"Started {self.max_workers} workers")

    async def stop_workers(self) -> None:
        """Stop all workers and wait for completion."""
        self.shutdown_event.set()
        logger.info("Shutting down workers")

        for worker in self.workers:
            worker.cancel()

        try:
            await asyncio.gather(*self.workers, return_exceptions=True)
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                raise
        self.workers.clear()

    async def worker(self, worker_name: str) -> None:
        """Background worker to process tasks."""
        logger.info(f"Starting worker {worker_name}")

        while not self.shutdown_event.is_set():
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                await self._process_task(task)
                self.queue.task_done()
            except TimeoutError:
                continue
            except Exception:
                logger.error(f"Worker {worker_name} error", exc_info=True)

    async def _process_task(self, task: dict[str, Any]) -> None:
        """Process individual task."""
        task_id = task["id"]
        task_type = task.get("task_type", "unknown")

        if task_id not in self.active_tasks:
            return

        self.active_tasks[task_id]["status"] = TaskStatus.RUNNING.value
        self.active_tasks[task_id]["started_at"] = pendulum.now("UTC").isoformat()
        self.active_tasks[task_id]["updated_at"] = pendulum.now("UTC").isoformat()

        try:
            processor = self.task_processors.get(task_type)
            if not processor:
                raise ValueError(f"No processor registered for task type: {task_type}")

            result = await processor(task)

            self.active_tasks[task_id]["status"] = TaskStatus.COMPLETED.value
            self.active_tasks[task_id]["result"] = result
            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            self.active_tasks[task_id]["status"] = TaskStatus.FAILED.value
            self.active_tasks[task_id]["error"] = str(e)
            logger.error(
                f"Task {task_id} failed", exc_info=True, extra={"task_id": task_id}
            )
        finally:
            now = pendulum.now("UTC").isoformat()
            self.active_tasks[task_id]["updated_at"] = now
            self.active_tasks[task_id]["completed_at"] = now


class SQSQueue:
    """AWS SQS implementation of QueueBackend - TODO: Implement during migration"""

    def __init__(self, sqs_settings: Any) -> None:
        # TODO: Implement SQS-based queue during cloud migration
        # Will use boto3 SQS client for queue operations
        # Workers will be deployed as separate ECS tasks/Lambda functions
        raise NotImplementedError(
            "SQS implementation will be added during cloud migration"
        )

    async def submit(self, task_type: TaskType, payload: dict) -> str:
        # TODO: Send message to SQS queue
        raise NotImplementedError(
            "SQS implementation will be added during cloud migration"
        )

    async def get_status(self, task_id: str) -> TaskInfo:
        # TODO: Query task status from DynamoDB or separate tracking system
        raise NotImplementedError(
            "SQS implementation will be added during cloud migration"
        )

    async def cancel(self, task_id: str) -> bool:
        # TODO: Remove message from SQS queue if possible
        raise NotImplementedError(
            "SQS implementation will be added during cloud migration"
        )


def get_queue(
    settings: Any, task_processors: dict[str, Callable] | None = None
) -> QueueBackend:
    """Get queue backend based on configuration"""
    # For now, always return InMemoryQueue
    # Later: return SQSQueue(settings.sqs) if settings.sqs.enabled
    return InMemoryQueue(
        max_workers=getattr(settings, "task_workers", 2),
        task_processors=task_processors or {},
    )
