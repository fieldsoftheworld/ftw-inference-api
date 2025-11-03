"""S2 Scene caching service for improved performance."""

import hashlib
import shutil
from pathlib import Path

import aiofiles.os

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_year_from_window(win_a: str) -> str:
    """Extract year from window string for partitioning."""
    try:
        first_date = win_a.split("_")[0]
        year = first_date.split("-")[0]
        # Validate it's actually a year (4 digits)
        if len(year) == 4 and year.isdigit():
            return year
        logger.warning(
            f"Could not extract valid year from win_a: {win_a}, using 'unknown'"
        )
        return "unknown"
    except (IndexError, AttributeError):
        logger.warning(f"Could not extract year from win_a: {win_a}, using 'unknown'")
        return "unknown"


def generate_scene_id(win_a: str, win_b: str, bbox: list[float]) -> str:
    """Generate unique scene ID from parameters."""
    scene_params = f"{win_a}|{win_b}|{','.join(map(str, bbox))}"
    scene_hash = hashlib.sha256(scene_params.encode()).hexdigest()[:12]
    year = extract_year_from_window(win_a)
    return f"{year}_{scene_hash}"


def get_cache_dir() -> Path:
    """Get the cache directory path from settings."""
    settings = get_settings()
    cache_dir = Path(settings.storage.output_dir).parent / "cache" / "scenes"
    return cache_dir


def get_scene_cache_path(win_a: str, win_b: str, bbox: list[float]) -> Path:
    """Get the full cache path for a scene, organized by year."""
    cache_dir = get_cache_dir()
    year = extract_year_from_window(win_a)
    scene_id = generate_scene_id(win_a, win_b, bbox)
    year_dir = cache_dir / year
    cache_path = year_dir / f"scene_{scene_id}.tif"
    return cache_path


async def check_s2_scene_exists(
    win_a: str,
    win_b: str,
    bbox: list[float],
) -> tuple[bool, Path | None]:
    """Check if S2 scene is already cached locally."""
    cache_path = get_scene_cache_path(win_a, win_b, bbox)

    if await aiofiles.os.path.exists(cache_path):
        logger.info(f"Cache HIT: Found cached scene at {cache_path}")
        return True, cache_path

    logger.info(f"Cache MISS: Scene not cached (would be at {cache_path})")
    return False, None


async def save_to_cache(
    image_file: Path,
    win_a: str,
    win_b: str,
    bbox: list[float],
) -> None:
    """Save downloaded image to cache for future use."""
    cache_path = get_scene_cache_path(win_a, win_b, bbox)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(image_file, cache_path)
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Saved scene to cache: {cache_path} ({size_mb:.2f} MB)",
            extra={"cache_size_mb": size_mb, "cache_path": str(cache_path)},
        )
    except Exception as e:
        logger.error(f"Failed to save scene to cache: {e}", exc_info=True)
