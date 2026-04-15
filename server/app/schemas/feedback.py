from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TileRatingRequest(BaseModel):
    """Request body for the tile-rating endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    rating: Literal[1, 2, 3] = Field(
        ..., description="Quality rating: 1=Not Great, 2=Ok, 3=Great!"
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="WGS84 viewport bbox [minLng, minLat, maxLng, maxLat]",
    )
    resolution: float = Field(
        ...,
        ge=0,
        description=(
            "Pixel resolution in meters at time of rating. "
            "Indicates how closely the user was inspecting the data."
        ),
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        min_lng, min_lat, max_lng, max_lat = v
        if min_lng > max_lng:
            raise ValueError("bbox minLng must be <= maxLng")
        if min_lat > max_lat:
            raise ValueError("bbox minLat must be <= maxLat")
        return v


class TileRatingResponse(BaseModel):
    """Response for the tile-rating endpoint."""

    rating_id: str = Field(..., description="UUID of the stored rating record")
    status: Literal["created", "updated"] = Field(
        ...,
        description=(
            "'created' if a new record was stored, "
            "'updated' if dedup logic was applied to an existing record"
        ),
    )


class TellUsMoreRequest(BaseModel):
    """Request body for the tell-us-more endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    quality_feedback: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="What was good or bad, and how must field boundaries improve?",
    )
    use_case: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="How would you use field boundaries? Describe your use case.",
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="WGS84 viewport bbox [minLng, minLat, maxLng, maxLat]",
    )
    resolution: float = Field(
        ...,
        ge=0,
        description=(
            "Pixel resolution in meters at time of submission. "
            "Indicates how closely the user was inspecting the data."
        ),
    )
    rating: Literal[1, 2, 3] | None = Field(
        default=None,
        description="Optional rating carried over from the tile-rating form",
    )
    name: str | None = Field(
        default=None, max_length=200, description="Optional submitter name"
    )
    email: str | None = Field(
        default=None, max_length=254, description="Optional submitter email address"
    )
    organization: str | None = Field(
        default=None, max_length=200, description="Optional submitter organization"
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        min_lng, min_lat, max_lng, max_lat = v
        if min_lng > max_lng:
            raise ValueError("bbox minLng must be <= maxLng")
        if min_lat > max_lat:
            raise ValueError("bbox minLat must be <= maxLat")
        return v

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str | None) -> str | None:
        if v is not None and "@" not in v:
            raise ValueError("Invalid email address")
        return v


class TellUsMoreResponse(BaseModel):
    """Response for the tell-us-more endpoint."""

    feedback_id: str = Field(..., description="UUID of the stored feedback record")
    status: Literal["submitted"]


class ContributeRequest(BaseModel):
    """Request body for the contribute endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    contribution_types: list[
        Literal["annotator", "share_data", "provide_models", "contribute_code"]
    ] = Field(
        ...,
        min_length=1,
        description=(
            "One or more ways the submitter would like to contribute: "
            "'annotator', 'share_data', 'provide_models', 'contribute_code'"
        ),
    )
    name: str = Field(
        ..., min_length=1, max_length=200, description="Required submitter name"
    )
    email: str = Field(
        ..., max_length=254, description="Required submitter email address"
    )
    resources: str | None = Field(
        default=None,
        max_length=5000,
        description=(
            "Resources (field boundaries, models, etc.) the user wants to share — "
            "for example, a dataset URL or model repository with a description."
        ),
    )
    additional_info: str | None = Field(
        default=None,
        max_length=5000,
        description="Anything else the submitter wants to share",
    )
    organization: str | None = Field(
        default=None, max_length=200, description="Optional submitter organization"
    )

    @field_validator("contribution_types")
    @classmethod
    def validate_unique_types(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("contribution_types must not contain duplicates")
        return v

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email address")
        return v


class ContributeResponse(BaseModel):
    """Response for the contribute endpoint."""

    contribution_id: str = Field(
        ..., description="UUID of the stored contribution record"
    )
    status: Literal["submitted"]


class AreaSummaryResponse(BaseModel):
    """Response for the area-summary endpoint."""

    bbox: list[float] = Field(
        ..., description="The requested WGS84 bbox [minLng, minLat, maxLng, maxLat]"
    )
    total_ratings: int = Field(
        ..., ge=0, description="Total ratings whose viewport intersects this bbox"
    )
    average_rating: float = Field(
        ...,
        ge=1.0,
        le=3.0,
        description="Average rating across all matching submissions",
    )
    rating_distribution: list[dict[str, int]] = Field(
        ...,
        description=(
            "Count of ratings for each value: "
            '[{"level": 1, "count": n}, {"level": 2, "count": n}, '
            '{"level": 3, "count": n}]'
        ),
    )
