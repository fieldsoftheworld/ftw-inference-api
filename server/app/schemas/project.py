from enum import Enum

import pendulum
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
)

from app.core.types import PendulumDateTime


class ProjectStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PolygonizeOptions(BaseModel):
    simplify: float = Field(
        15, description="Simplification factor to use when polygonizing"
    )
    min_size: float = Field(
        500, description="Minimum area size in square meters to include in the output"
    )
    close_interiors: bool = Field(
        False, description="Remove the interiors holes in the polygons if set to `true`"
    )


class InferenceParameters(BaseModel):
    queue: bool = Field(
        True, description="Whether to queue the inference or run it immediately"
    )
    model: str = Field(..., description="The id of the model to use for inference")
    images: list[str] | None = Field(
        None, description="A list of two publicly accessible image URLs"
    )
    resize_factor: float = Field(2, description="Resize factor to use for inference")
    patch_size: int = Field(1024, description="Size of patch to use for inference")
    padding: int = Field(
        64, description="Pixels to discard from each side of the patch"
    )
    polygonize: PolygonizeOptions | None = Field(
        None, description="Options for polygonization"
    )


class ProjectCreate(BaseModel):
    title: str


class ProjectResponse(BaseModel):
    id: str
    title: str
    status: ProjectStatus
    progress: float | None = None
    created_at: PendulumDateTime
    parameters: InferenceParameters | None = None

    model_config = ConfigDict(
        from_attributes=True,  # Replaces orm_mode=True
    )

    @field_serializer("created_at")
    def serialize_datetime(self, dt: pendulum.DateTime) -> str:
        """Serialize pendulum DateTime to ISO8601 string with Z timezone designator"""
        if isinstance(dt, pendulum.DateTime):
            return dt.in_timezone("UTC").isoformat().replace("+00:00", "Z")
        return str(dt)


class ProjectsResponse(BaseModel):
    projects: list[ProjectResponse]


class ErrorResponse(BaseModel):
    error: str


class ModelInfo(BaseModel):
    id: str
    title: str
    description: str
    license: str
    version: str


class RootResponse(BaseModel):
    api_version: str
    title: str
    description: str
    models: list[ModelInfo]
