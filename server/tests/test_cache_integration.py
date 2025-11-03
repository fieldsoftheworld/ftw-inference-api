"""Integration tests for S2 scene caching."""

import shutil
import tempfile
from pathlib import Path

import pytest

from app.services.cache_service import (
    check_s2_scene_exists,
    get_cache_dir,
    save_to_cache,
)


@pytest.fixture
def temp_cache_dir(monkeypatch):
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock get_cache_dir to use our temp directory
        temp_path = Path(tmpdir) / "cache" / "scenes"
        monkeypatch.setattr(
            "app.services.cache_service.get_cache_dir", lambda: temp_path
        )
        yield temp_path


@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        f.write(b"fake image data for testing")
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestCacheIntegration:
    """Integration tests for cache functionality."""

    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, temp_cache_dir, sample_image_file):
        """Test that cache miss works, then cache hit after saving."""
        win_a = "2023-01-01_2023-03-31"
        win_b = "2023-04-01_2023-06-30"
        bbox = [10.0, 20.0, 10.5, 20.5]

        # First check - should be a cache MISS
        exists, path = await check_s2_scene_exists(win_a, win_b, bbox)
        assert exists is False
        assert path is None

        # Save to cache
        await save_to_cache(sample_image_file, win_a, win_b, bbox)

        # Second check - should be a cache HIT
        exists, path = await check_s2_scene_exists(win_a, win_b, bbox)
        assert exists is True
        assert path is not None
        assert path.exists()
        assert path.suffix == ".tif"

    @pytest.mark.asyncio
    async def test_cache_organizes_by_year(self, temp_cache_dir, sample_image_file):
        """Test that cache files are organized in year directories."""
        win_a = "2023-01-01_2023-03-31"
        win_b = "2023-04-01_2023-06-30"
        bbox = [10.0, 20.0, 10.5, 20.5]

        await save_to_cache(sample_image_file, win_a, win_b, bbox)

        # Check that year directory was created
        year_dir = temp_cache_dir / "2023"
        assert year_dir.exists()
        assert year_dir.is_dir()

        # Check that file is in the year directory
        cached_files = list(year_dir.glob("scene_*.tif"))
        assert len(cached_files) == 1

    @pytest.mark.asyncio
    async def test_multiple_scenes_cached(self, temp_cache_dir, sample_image_file):
        """Test that multiple different scenes can be cached."""
        scenes = [
            (
                "2023-01-01_2023-03-31",
                "2023-04-01_2023-06-30",
                [10.0, 20.0, 10.5, 20.5],
            ),
            (
                "2023-07-01_2023-09-30",
                "2023-10-01_2023-12-31",
                [11.0, 21.0, 11.5, 21.5],
            ),
            (
                "2024-01-01_2024-03-31",
                "2024-04-01_2024-06-30",
                [12.0, 22.0, 12.5, 22.5],
            ),
        ]

        # Cache all scenes
        for win_a, win_b, bbox in scenes:
            await save_to_cache(sample_image_file, win_a, win_b, bbox)

        # Verify all can be found
        for win_a, win_b, bbox in scenes:
            exists, path = await check_s2_scene_exists(win_a, win_b, bbox)
            assert exists is True
            assert path is not None

    @pytest.mark.asyncio
    async def test_cached_file_content_preserved(
        self, temp_cache_dir, sample_image_file
    ):
        """Test that cached file content is preserved correctly."""
        win_a = "2023-01-01_2023-03-31"
        win_b = "2023-04-01_2023-06-30"
        bbox = [10.0, 20.0, 10.5, 20.5]

        # Read original content
        with open(sample_image_file, "rb") as f:
            original_content = f.read()

        # Save to cache
        await save_to_cache(sample_image_file, win_a, win_b, bbox)

        # Get cached file
        exists, cached_path = await check_s2_scene_exists(win_a, win_b, bbox)
        assert exists is True

        # Verify content matches
        with open(cached_path, "rb") as f:
            cached_content = f.read()

        assert cached_content == original_content

    @pytest.mark.asyncio
    async def test_different_bbox_different_cache(
        self, temp_cache_dir, sample_image_file
    ):
        """Test that different bboxes create different cache entries."""
        win_a = "2023-01-01_2023-03-31"
        win_b = "2023-04-01_2023-06-30"
        bbox1 = [10.0, 20.0, 10.5, 20.5]
        bbox2 = [10.0, 20.0, 10.6, 20.6]  # Slightly different

        # Cache first bbox
        await save_to_cache(sample_image_file, win_a, win_b, bbox1)

        # First bbox should hit
        exists1, _ = await check_s2_scene_exists(win_a, win_b, bbox1)
        assert exists1 is True

        # Second bbox should miss
        exists2, _ = await check_s2_scene_exists(win_a, win_b, bbox2)
        assert exists2 is False
