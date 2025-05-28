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


class PolygonizationParameters(BaseModel):
    simplify: float = Field(
        15,
        description="Simplification factor to use when polygonizing in the "
        + "unit of the CRS",
    )
    min_size: float = Field(
        500, description="Minimum area size in square meters to include in the output"
    )
    close_interiors: bool = Field(
        False, description="Remove the interiors holes in the polygons if set to `true`"
    )


class InferenceParameters(BaseModel):
    model: str = Field(..., description="The id of the model to use for inference")
    bbox: list[float] | None = Field(
        None,
        description="The bounding box of the area to run inference on "
        + "[minX, minY, maxX, maxY]",
    )
    images: list[str] | None = Field(
        None,
        description="A list of two publicly accessible image URLs "
        + "(window A and B) to run inference on",
    )
    resize_factor: float = Field(2, description="Resize factor to use for inference")
    patch_size: int | None = Field(None, description="Size of patch to use for inference")
    padding: int = Field(
        0, description="Pixels to discard from each side of the patch"
    )


class ProcessingParameters(BaseModel):
    inference: InferenceParameters | None = Field(
        default=None,
        description="Parameters for running inference on the project",
    )
    polygons: PolygonizationParameters | None = Field(
        default=None,
        description="Parameters for polygonization of the inference results",
    )

    model_config = ConfigDict(
        extra="ignore",  # Ignore additional fields
    )


class ProjectResults(BaseModel):
    inference: str | None = Field(
        default=None,
        description="The (signed) URL to the inference results. "
        + "Content type is image/tiff; application=geotiff; cloud-optimized=true.",
    )
    polygons: str | None = Field(
        default=None,
        description="The (signed) URL to the polygons. "
        + "Content type is application/geo+json.",
    )


class ProjectCreate(BaseModel):
    title: str


class ProjectResponse(BaseModel):
    id: str
    title: str
    status: ProjectStatus
    progress: float | None = None
    created_at: PendulumDateTime
    parameters: ProcessingParameters = Field(
        default_factory=ProcessingParameters,
        description="Parameters for processing the project",
    )
    results: ProjectResults = Field(
        default_factory=ProjectResults,
        description="Results of inference and polygonization processing",
    )

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
    max_area_km2: float
    models: list[ModelInfo]
