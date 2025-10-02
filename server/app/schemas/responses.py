from typing import Any, Literal

import pendulum
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
)

from app.core.types import PendulumDateTime, ProjectStatus
from app.schemas.parameters import ModelInfo, ProjectResultLinks, TaskInfo


class RootResponse(BaseModel):
    """Response for the root endpoint, providing API information."""

    api_version: str
    title: str
    description: str
    min_area_km2: float
    example_max_area_km2: float
    project_max_area_km2: float
    example_endpoint_enabled: bool
    models: list[ModelInfo]


class ProjectResponse(BaseModel):
    """Response model for a single project's details."""

    id: str
    title: str
    status: ProjectStatus
    progress: float | None = None
    created_at: PendulumDateTime
    parameters: dict = Field(
        default_factory=dict,
        description="Parameters for processing the project",
    )
    results: ProjectResultLinks = Field(
        default_factory=ProjectResultLinks,
        description="Results of inference and polygonization processing",
    )

    model_config = ConfigDict(
        from_attributes=True,
    )

    @field_serializer("created_at")
    def serialize_datetime(self, dt: pendulum.DateTime) -> str:
        """Serialize pendulum DateTime to ISO8601 string with Z timezone designator"""
        if isinstance(dt, pendulum.DateTime):
            iso_str: str = dt.in_timezone("UTC").isoformat().replace("+00:00", "Z")
            return iso_str
        return str(dt) if dt is not None else ""


class ProjectsResponse(BaseModel):
    """Response model for a list of projects."""

    projects: list[ProjectResponse]


class ErrorResponse(BaseModel):
    """Generic error response model."""

    error: str


class TaskSubmissionResponse(BaseModel):
    """Response when a task is successfully submitted."""

    message: str
    task_id: str
    project_id: str
    status: Literal["queued"]


class ProjectStatusResponse(BaseModel):
    """Complete project status with task information."""

    project_id: str
    status: str
    progress: int | None = None
    parameters: dict[str, Any]
    task: TaskInfo | None = None
    polygonize_task: TaskInfo | None = None


class TaskDetailsResponse(BaseModel):
    """Response for individual task details."""

    task_id: str
    task_type: str
    status: str
    project_id: str
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class InferenceResultsResponse(BaseModel):
    """Response model for inference results with URLs."""

    inference: str | None = Field(None, description="URL to inference TIF result")
    polygons: str | None = Field(None, description="URL to GeoJSON polygons result")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"]
