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
    max_area_km2: float
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

    @field_serializer("parameters")
    def serialize_parameters(self, parameters: Any) -> dict[str, Any]:
        """Clean parameters for API response, excluding large fields."""
        if isinstance(parameters, dict):
            params_dict = parameters
        else:
            params_dict = (
                parameters.model_dump() if hasattr(parameters, "model_dump") else {}
            )

        if not params_dict:
            return {}

        clean_params = {}

        if "inference" in params_dict:
            inf_params = params_dict["inference"]
            if inf_params:
                clean_params["inference"] = {
                    k: v for k, v in inf_params.items() if k != "images"
                }

                if "model" in inf_params:
                    model_value = inf_params["model"]
                    if isinstance(model_value, str) and model_value.endswith(".ckpt"):
                        clean_params["inference"]["model"] = model_value.split("/")[
                            -1
                        ].replace(".ckpt", "")
                    else:
                        clean_params["inference"]["model"] = model_value

                if "images" in inf_params:
                    clean_params["inference"]["images_count"] = len(
                        inf_params["images"]
                    )

        for key in ["polygons", "task_id", "polygonize_task_id"]:
            if key in params_dict:
                clean_params[key] = params_dict[key]

        return clean_params


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


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"]
