"""Tests for cache service functionality."""

import tempfile
from pathlib import Path

import pytest

from app.services.cache_service import (
    extract_year_from_window,
    generate_scene_id,
    get_scene_cache_path,
)


class TestCacheService:
    """Test cache service functions."""

    def test_extract_year_from_window(self):
        """Test year extraction from window string."""
        assert extract_year_from_window("2023-01-01_2023-03-31") == "2023"
        assert extract_year_from_window("2024-06-01_2024-08-31") == "2024"
        assert extract_year_from_window("2022-12-01_2023-02-28") == "2022"

    def test_extract_year_from_invalid_window(self):
        """Test year extraction with invalid input."""
        assert extract_year_from_window("invalid") == "unknown"
        assert extract_year_from_window("") == "unknown"

    def test_generate_scene_id(self):
        """Test scene ID generation."""
        win_a = "2023-01-01_2023-03-31"
        win_b = "2023-04-01_2023-06-30"
        bbox = [10.0, 20.0, 10.5, 20.5]

        scene_id = generate_scene_id(win_a, win_b, bbox)

        # Should start with year
        assert scene_id.startswith("2023_")
        # Should have a hash component
        assert len(scene_id) > 5

    def test_generate_scene_id_deterministic(self):
        """Test that same inputs generate same scene ID."""
        win_a = "2023-01-01_2023-03-31"
        win_b = "2023-04-01_2023-06-30"
        bbox = [10.0, 20.0, 10.5, 20.5]

        scene_id1 = generate_scene_id(win_a, win_b, bbox)
        scene_id2 = generate_scene_id(win_a, win_b, bbox)

        assert scene_id1 == scene_id2

    def test_generate_scene_id_different_inputs(self):
        """Test that different inputs generate different scene IDs."""
        win_a = "2023-01-01_2023-03-31"
        win_b1 = "2023-04-01_2023-06-30"
        win_b2 = "2023-07-01_2023-09-30"
        bbox = [10.0, 20.0, 10.5, 20.5]

        scene_id1 = generate_scene_id(win_a, win_b1, bbox)
        scene_id2 = generate_scene_id(win_a, win_b2, bbox)

        assert scene_id1 != scene_id2

    def test_get_scene_cache_path(self):
        """Test cache path generation."""
        win_a = "2023-01-01_2023-03-31"
        win_b = "2023-04-01_2023-06-30"
        bbox = [10.0, 20.0, 10.5, 20.5]

        cache_path = get_scene_cache_path(win_a, win_b, bbox)

        # Should be a Path object
        assert isinstance(cache_path, Path)
        # Should contain year directory
        assert "2023" in str(cache_path)
        # Should be a .tif file
        assert cache_path.suffix == ".tif"
        # Should contain "scene_" prefix
        assert "scene_" in cache_path.name
