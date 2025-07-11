from enum import Enum
from typing import Literal

import pendulum
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
)

from app.core.config import get_settings
from app.core.types import PendulumDateTime

allowed_models = [model.get("id") for model in get_settings().models]


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
    model: Literal[tuple(allowed_models)] = Field(  # type: ignore
        ..., description="The id of the model to use for inference"
    )
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
    resize_factor: int = Field(2, description="Resize factor to use for inference")
    patch_size: int | None = Field(
        None, description="Size of patch to use for inference"
    )
    padding: int | None = Field(
        None, description="Pixels to discard from each side of the patch"
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
    parameters: dict = Field(
        default_factory=dict,
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

    @field_serializer("parameters")
    def serialize_parameters(self, parameters) -> dict:
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

        # Clean inference parameters
        if "inference" in params_dict:
            inf_params = params_dict["inference"]
            if inf_params:
                clean_params["inference"] = {
                    k: v
                    for k, v in inf_params.items()
                    if k != "images"  # Exclude large images array
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

        # Copy other parameters as-is
        for key in ["polygons", "task_id", "polygonize_task_id"]:
            if key in params_dict:
                clean_params[key] = params_dict[key]

        return clean_params


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
    min_area_km2: float
    max_area_km2: float
    models: list[ModelInfo]
