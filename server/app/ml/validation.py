"""ML parameter validation functions."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.core.config import get_settings
from app.core.geo import calculate_area_km2


def validate_bbox(
    bbox: Any, require_bbox: bool = False, max_area: float | None = None
) -> None:
    """Validate bounding box coordinates and area constraints."""
    if require_bbox and not isinstance(bbox, list):
        raise ValueError("Bounding box is required as a list of four values.")

    if not isinstance(bbox, list):
        return

    if len(bbox) != 4:
        raise ValueError("Bounding box must be in format [minX, minY, maxX, maxY]")

    min_lon, min_lat, max_lon, max_lat = bbox

    if min_lon < -180 or max_lon > 180:
        raise ValueError(
            "Longitude values must be between -180 and 180 degrees in EPSG:4326"
        )

    if min_lat < -90 or max_lat > 90:
        raise ValueError(
            "Latitude values must be between -90 and 90 degrees in EPSG:4326"
        )

    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(
            "Invalid bounding box: min values must be less than max values"
        )

    if max_area is not None:
        area_size_km2 = calculate_area_km2(bbox)
        if area_size_km2 > max_area:
            raise ValueError(
                f"Area too large for example endpoint. "
                f"Maximum allowed area is {max_area} km². "
                "Please create a project instead."
            )


def validate_image_urls(urls: Any, require_image_urls: bool = False) -> None:
    """Validate image URLs format and structure."""
    if not require_image_urls:
        return

    if not isinstance(urls, list) or len(urls) != 2:
        raise ValueError("Images must be a list of two items")

    matcher = re.compile(r"^https?://[\w/.-]+$", re.A | re.I)
    for url in urls:
        if not matcher.match(url):
            raise ValueError(f"URL '{url}' contains invalid characters")
        result = urlparse(url)
        if not result.scheme or not result.netloc or not result.path:
            raise ValueError(f"URL '{url}' is invalid")


def validate_model(params: dict[str, Any]) -> None:
    """Validate model configuration and file existence."""
    settings = get_settings()
    model_id = params.get("model")

    model_config = next(
        (model for model in settings.models if model.get("id") == model_id),
        None,
    )
    if not model_config:
        raise ValueError(f"Model with ID '{model_id}' not found")

    model_file = model_config.get("file")
    if not model_file:
        raise ValueError(f"Model '{model_id}' has no file specified")

    model_path = Path(__file__).parent.parent.parent / "data" / "models" / model_file
    if not model_path.exists():
        raise ValueError(f"Model file not found at '{model_path}'")


def resolve_model_path(model_id: str) -> str:
    """Resolve model ID to absolute file path for ML execution."""
    settings = get_settings()

    model_config = next(
        (model for model in settings.models if model.get("id") == model_id),
        None,
    )
    if not model_config:
        raise ValueError(f"Model with ID '{model_id}' not found")

    model_file = model_config.get("file")
    if not model_file:
        raise ValueError(f"Model '{model_id}' has no file specified")

    model_path = Path(__file__).parent.parent.parent / "data" / "models" / model_file
    return str(model_path.absolute())


def validate_processing_params(params: dict[str, Any]) -> None:
    """Validate ML processing parameters."""
    if params.get("resize_factor", 1) <= 0:
        raise ValueError("Resize factor must be a positive number")

    padding = params.get("padding")
    if padding is not None and padding < 0:
        raise ValueError("Padding must be null, a positive integer or 0")

    patch_size = params.get("patch_size")
    if patch_size is not None and patch_size % 32 != 0:
        raise ValueError("Patch size must be a multiple of 32.")


def validate_year(year: int) -> None:
    """Validate year is within reasonable range for Sentinel-2 data."""
    current_year = datetime.now().year
    if year < 2015:
        raise ValueError("Year must be 2015 or later (Sentinel-2 launch year)")
    if year > current_year:
        raise ValueError(f"Year must be {current_year} or earlier")


def validate_cloud_cover(cloud_cover: int) -> None:
    """Validate cloud cover percentage is between 0-100."""
    if cloud_cover < 0 or cloud_cover > 100:
        raise ValueError("Cloud cover must be between 0 and 100 percent")


def validate_buffer_days(buffer_days: int) -> None:
    """Validate buffer days is reasonable positive integer."""
    if buffer_days < 0:
        raise ValueError("Buffer days must be 0 or positive")
    if buffer_days > 365:
        raise ValueError("Buffer days must be 365 or less")


def prepare_scene_selection_params(params: dict[str, Any]) -> dict[str, Any]:
    """Prepare and validate scene selection parameters."""
    validate_bbox(params.get("bbox"), require_bbox=True)
    year = params.get("year")
    if year is not None:
        validate_year(year)
    validate_cloud_cover(params.get("cloud_cover_max", 20))
    validate_buffer_days(params.get("buffer_days", 14))
    return params


def prepare_inference_params(
    params: dict[str, Any],
    require_bbox: bool = False,
    max_area: float | None = None,
    require_image_urls: bool = False,
) -> dict[str, Any]:
    """Prepare and validate inference parameters."""
    validate_bbox(params.get("bbox"), require_bbox, max_area)
    validate_image_urls(params.get("images"), require_image_urls)
    validate_model(params)
    validate_processing_params(params)
    return params
