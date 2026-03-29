from typing import Literal

from ftw_tools.inference.model_registry import MODEL_REGISTRY
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.logging import get_logger

allowed_models = list(MODEL_REGISTRY.keys()) or ["default_model"]
logger = get_logger(__name__)


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

    @field_validator("model")
    @classmethod
    def validate_model_exists(cls, v: str) -> str:
        """Ensure model exists in registry."""
        if v not in MODEL_REGISTRY:
            available = ", ".join(list(MODEL_REGISTRY.keys())[:3]) + "..."
            raise ValueError(
                f"Unknown model '{v}'. Available models include: {available}. "
                f"Use GET /v1/models to see all available models."
            )
        return v

    @model_validator(mode="after")
    def validate_image_count_for_model(self) -> "InferenceRequest":
        """Validate image count matches model requirements."""
        if self.images is None:
            return self  # Skip validation if no images provided

        model_spec = MODEL_REGISTRY.get(self.model)
        if not model_spec:
            return self  # Model validation handled above

        expected_count = 2 if model_spec.requires_window else 1
        actual_count = len(self.images)

        if actual_count != expected_count:
            window_type = (
                "temporal windows" if model_spec.requires_window else "single window"
            )
            raise ValueError(
                f"Model '{self.model}' requires exactly {expected_count} "
                f"image(s) ({window_type}), but {actual_count} provided"
            )
        return self


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

    @model_validator(mode="after")
    def validate_polygonization_compatibility(self) -> "ExampleWorkflowRequest":
        """Warn if polygonization requested for models that don't need it."""
        if not self.inference or not self.polygons:
            return self

        model_spec = MODEL_REGISTRY.get(self.inference.model)
        if model_spec and not model_spec.requires_polygonize:
            # Just log warning, don't fail - user might want raw output anyway
            logger.warning(
                f"Model '{self.inference.model}' outputs GeoJSON directly. "
                f"Polygonization parameters will be ignored."
            )
        return self


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


class BenchmarkRunRequest(BaseModel):
    """Start an FTW benchmark run against dataset chips (official models only)."""

    model_ids: list[str] = Field(
        ...,
        min_length=1,
        description="One or more model ids from GET /v1/models",
    )
    country_ids: list[str] = Field(
        ...,
        min_length=1,
        description="FTW benchmark country ids from GET /v1/benchmarks/countries",
    )
    split: Literal["train", "validation", "val", "test"] = Field(
        default="test",
        description="Chip split to evaluate (default test)",
    )
    max_chips: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Maximum chips per country per model (cost guard)",
    )
    seed: int | None = Field(default=None, description="Random seed for chip subsample")
    iou_threshold: float = Field(
        default=0.25,
        ge=0.1,
        le=0.99,
        description=(
            "IoU threshold for greedy pred↔GT instance matching. "
            "Auto-downloaded STAC is not the benchmark’s original paired imagery, "
            "so overlaps are often below 0.5; 0.25 is a practical default. "
            "Use ~0.5 when evaluating the full local FTW release."
        ),
    )
    include_map_geojson: bool = Field(
        default=True,
        description=(
            "If true, the result may include map_geojson (chip footprints plus GT and "
            "prediction polygons) when max_chips is at most 40. Omit or set false to "
            "reduce response size."
        ),
    )

    @field_validator("model_ids")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        for mid in v:
            if mid not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model '{mid}'. Use GET /v1/models for valid ids."
                )
        return v
