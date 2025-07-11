from dataclasses import dataclass, field
from typing import Any

from app.core.types import TaskType


@dataclass
class Task:
    """Base task class with common fields."""

    id: str
    task_type: TaskType
    project_id: str
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "project_id": self.project_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        return cls(
            id=data["id"],
            task_type=TaskType(data["task_type"]),
            project_id=data["project_id"],
            status=data.get("status", "pending"),
        )

    def validate(self) -> None:
        """Validate task data."""
        if not self.id:
            raise ValueError("Task ID is required")
        if not self.project_id:
            raise ValueError("Project ID is required")
        if not isinstance(self.task_type, TaskType):
            raise ValueError("Task type must be a valid TaskType enum")


@dataclass
class InferenceTask(Task):
    """Task for running ML inference on project images."""

    inference_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set task type after initialization."""
        self.task_type = TaskType.INFERENCE

    def to_dict(self) -> dict[str, Any]:
        """Convert inference task to dictionary."""
        base_dict = super().to_dict()
        base_dict["inference_params"] = self.inference_params
        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceTask":
        """Create inference task from dictionary."""
        return cls(
            id=data["id"],
            task_type=TaskType(data["task_type"]),
            project_id=data["project_id"],
            status=data.get("status", "pending"),
            inference_params=data["inference_params"],
        )

    def validate(self) -> None:
        """Validate inference task data."""
        super().validate()
        if not self.inference_params:
            raise ValueError("Inference parameters are required")

        # Validate required inference parameters
        required_params = ["model", "resize_factor"]
        for param in required_params:
            if param not in self.inference_params:
                raise ValueError(f"Missing required inference parameter: {param}")


@dataclass
class PolygonizeTask(Task):
    """Task for generating polygons from inference results."""

    polygon_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set task type after initialization."""
        self.task_type = TaskType.POLYGONIZE

    def to_dict(self) -> dict[str, Any]:
        """Convert polygonize task to dictionary."""
        base_dict = super().to_dict()
        base_dict["polygon_params"] = self.polygon_params
        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolygonizeTask":
        """Create polygonize task from dictionary."""
        return cls(
            id=data["id"],
            task_type=TaskType(data["task_type"]),
            project_id=data["project_id"],
            status=data.get("status", "pending"),
            polygon_params=data["polygon_params"],
        )

    def validate(self) -> None:
        """Validate polygonize task data."""
        super().validate()
        if not self.polygon_params:
            raise ValueError("Polygon parameters are required")

        # Validate required polygon parameters
        required_params = ["simplify", "min_size", "close_interiors"]
        for param in required_params:
            if param not in self.polygon_params:
                raise ValueError(f"Missing required polygon parameter: {param}")


def create_task(task_type: TaskType, project_id: str, **kwargs) -> Task:
    """Factory function to create tasks of the appropriate type."""
    if task_type == TaskType.INFERENCE:
        if "inference_params" not in kwargs:
            raise ValueError("Inference parameters are required for inference tasks")
        return InferenceTask(
            id="",  # Will be set by task manager
            task_type=task_type,
            project_id=project_id,
            inference_params=kwargs["inference_params"],
        )
    elif task_type == TaskType.POLYGONIZE:
        if "polygon_params" not in kwargs:
            raise ValueError("Polygon parameters are required for polygonize tasks")
        return PolygonizeTask(
            id="",  # Will be set by task manager
            task_type=task_type,
            project_id=project_id,
            polygon_params=kwargs["polygon_params"],
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def task_from_dict(data: dict[str, Any]) -> Task:
    """Create task from dictionary based on task type."""
    task_type = TaskType(data["task_type"])

    if task_type == TaskType.INFERENCE:
        return InferenceTask.from_dict(data)
    elif task_type == TaskType.POLYGONIZE:
        return PolygonizeTask.from_dict(data)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
