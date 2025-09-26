from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings

# This logic is used for validation and is acceptable at the schema definition level.
allowed_models = [model.get("id") for model in get_settings().models] or [
    "default_model"
]


class PolygonizationRequest(BaseModel):
    """Parameters for the polygonization process."""

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


class InferenceRequest(BaseModel):
    """Parameters for the inference process."""

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


class ExampleWorkflowRequest(BaseModel):
    """Parameters for running the example workflow."""

    inference: InferenceRequest | None = Field(
        default=None,
        description="Parameters for running inference on the project",
    )
    polygons: PolygonizationRequest | None = Field(
        default=None,
        description="Parameters for polygonization of the inference results",
    )

    model_config = ConfigDict(
        extra="ignore",  # Ignore additional fields
    )


class SceneSelectionRequest(BaseModel):
    """Parameters for Sentinel-2 scene selection."""

    year: int = Field(
        ..., description="Year for scene selection (2015 to current year)"
    )
    bbox: list[float] = Field(
        ..., description="Bounding box [minX, minY, maxX, maxY] in EPSG:4326"
    )
    cloud_cover_max: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum cloud cover percentage (0-100). Default: 20",
    )
    buffer_days: int = Field(
        default=14,
        ge=0,
        le=365,
        description="Buffer days around target date. Default: 14",
    )
    stac_host: Literal["mspc", "earthsearch"] = Field(
        default="earthsearch",
        description=(
            "STAC API host to use. 'mspc' for Microsoft Planetary Computer, "
            "'earthsearch' for Earth Search. Default: 'earthsearch'"
        ),
    )
    s2_collection: str = Field(
        default="c1",
        description="Sentinel-2 collection version. Default: 'c1'",
    )


class CreateProjectRequest(BaseModel):
    """Request to create a new project."""

    title: str
