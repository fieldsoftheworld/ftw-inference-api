import asyncio
import uuid
from typing import Any

from app.core.types import TaskStatus
from app.db.database import get_db
from app.models.project import InferenceResult, Project


class TaskManager:
    def __init__(self, max_workers: int = 2) -> None:
        """Initialize task manager with worker pool."""
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self.active_tasks: dict[str, dict[str, Any]] = {}
        self.max_workers = max_workers
        self.shutdown_event = asyncio.Event()

    async def submit_task(self, task: dict[str, Any]) -> str:
        """Submit task to queue and return task ID."""
        task_id = str(uuid.uuid4())
        task_data = {"id": task_id, **task}
        self.active_tasks[task_id] = {
            "status": TaskStatus.PENDING.value,
            "task_type": task.get("task_type", "unknown"),
            "project_id": task.get("project_id"),
            "created_at": asyncio.get_event_loop().time(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
        }
        await self.queue.put(task_data)
        return task_id

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get current task status."""
        if task_id in self.active_tasks:
            return TaskStatus(self.active_tasks[task_id]["status"])
        return TaskStatus.PENDING

    async def get_task_info(self, task_id: str) -> dict[str, Any] | None:
        """Get complete task information."""
        return self.active_tasks.get(task_id)

    async def start_workers(self) -> None:
        """Start background workers."""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self.worker(f"worker-{i}"))
            self.workers.append(worker)

    async def stop_workers(self) -> None:
        """Stop all workers and wait for completion."""
        self.shutdown_event.set()

        for worker in self.workers:
            worker.cancel()

        # Wait for all workers to finish
        try:
            await asyncio.gather(*self.workers, return_exceptions=True)
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                raise
        self.workers.clear()

    async def worker(self, worker_name: str) -> None:
        """Background worker to process tasks."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for task with timeout to check shutdown periodically
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                await self._process_task(task)
                self.queue.task_done()
            except asyncio.TimeoutError:
                # Continue loop to check shutdown event
                continue
            except Exception as e:
                # Log error but continue processing other tasks
                print(f"Worker {worker_name} error: {e}")

    async def _process_task(self, task: dict[str, Any]) -> None:
        """Process individual task."""
        task_id = task["id"]
        task_type = task.get("task_type", "unknown")

        if task_id not in self.active_tasks:
            return

        self.active_tasks[task_id]["status"] = TaskStatus.RUNNING.value
        self.active_tasks[task_id]["started_at"] = asyncio.get_event_loop().time()

        try:
            if task_type == "inference":
                result = await self._process_inference_task(task)
            elif task_type == "polygonize":
                result = await self._process_polygonize_task(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            self.active_tasks[task_id]["status"] = TaskStatus.COMPLETED.value
            self.active_tasks[task_id]["result"] = result

        except Exception as e:
            self.active_tasks[task_id]["status"] = TaskStatus.FAILED.value
            self.active_tasks[task_id]["error"] = str(e)
            print(f"Task {task_id} failed: {e}")
        finally:
            self.active_tasks[task_id]["completed_at"] = asyncio.get_event_loop().time()

    @staticmethod
    async def _process_inference_task(task: dict[str, Any]) -> dict[str, Any]:
        """Process inference task."""
        from app.core.processing import run_project_inference

        project_id = task["project_id"]
        params = task["inference_params"]

        db = next(get_db())
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "running"
                db.commit()

            result = await run_project_inference(project_id, params)

            if project and result.get("inference_file"):
                inference_result = InferenceResult(
                    project_id=project_id,
                    model_id=params.get("model", "unknown"),
                    result_type="inference",
                    file_path=result["inference_file"],
                )
                db.add(inference_result)

                project.results["inference"] = {
                    "file_path": result["inference_file"],
                    "metrics": {
                        k: v for k, v in result.items() if k != "inference_file"
                    },
                }
                project.status = "completed"
                db.commit()

            return result
        finally:
            db.close()

    @staticmethod
    async def _process_polygonize_task(task: dict[str, Any]) -> dict[str, Any]:
        """Process polygonization task."""
        from app.core.processing import run_project_polygonize

        project_id = task["project_id"]
        params = task["polygon_params"]

        db = next(get_db())
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "running"
                db.commit()

            result = await run_project_polygonize(project_id, params)

            if project and result.get("polygon_file"):
                polygon_result = InferenceResult(
                    project_id=project_id,
                    model_id="polygonization",
                    result_type="geojson",
                    file_path=result["polygon_file"],
                )
                db.add(polygon_result)

                project.results["polygons"] = {
                    "file_path": result["polygon_file"],
                    "metrics": {k: v for k, v in result.items() if k != "polygon_file"},
                }
                project.status = "completed"
                db.commit()

            return result
        finally:
            db.close()

    async def cleanup_completed_tasks(self, max_age_seconds: float = 3600) -> None:
        """Clean up completed tasks older than max_age_seconds."""
        current_time = asyncio.get_event_loop().time()
        tasks_to_remove = []

        for task_id, task_info in self.active_tasks.items():
            completed_statuses = [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]
            if (
                task_info["status"] in completed_statuses
                and task_info["completed_at"] is not None
                and current_time - task_info["completed_at"] > max_age_seconds
            ):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]


# Global task manager instance
task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance."""
    global task_manager
    if task_manager is None:
        task_manager = TaskManager()
    return task_manager


async def initialize_task_manager() -> None:
    """Initialize and start the task manager."""
    manager = get_task_manager()
    await manager.start_workers()


async def shutdown_task_manager() -> None:
    """Shutdown the task manager."""
    global task_manager
    if task_manager is not None:
        await task_manager.stop_workers()
        task_manager = None
