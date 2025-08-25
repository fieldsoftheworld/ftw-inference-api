from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about an available ML model."""

    id: str
    title: str
    description: str
    license: str
    version: str
    requires_window: bool = Field(
        description="Whether this model requires window A and B images"
    )
    requires_polygonize: bool = Field(
        description="Whether this model requires polygonization step"
    )


class TaskInfo(BaseModel):
    """Task information embedded in project status."""

    task_id: str
    task_type: str
    task_status: str
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class ProjectResultLinks(BaseModel):
    """Links to the output results of a project."""

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
