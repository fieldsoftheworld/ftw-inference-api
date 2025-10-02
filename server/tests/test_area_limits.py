"""Tests for area limit validation."""

import pytest
from app.core.config import Settings
from app.ml.validation import validate_bbox_area


def test_validate_bbox_area_too_small():
    """Should reject areas smaller than 0.1 km²."""
    # Tiny box near the equator (~0.012 km²)
    bbox = [0.0, 0.0, 0.001, 0.001]
    with pytest.raises(ValueError, match="Minimum area is 0.1"):
        validate_bbox_area(bbox, is_project=True, settings=Settings())


def test_validate_bbox_area_example_valid():
    """Should accept a small example area under 500 km²."""
    # ~200 km² near 45° lat (0.15 x 0.15 degrees)
    bbox = [10.0, 45.0, 10.15, 45.15]
    area = validate_bbox_area(bbox, is_project=False, settings=Settings())
    assert 150 < area < 250


def test_validate_bbox_area_example_too_large():
    """Should reject example areas over 500 km²."""
    # ~8,600 km² near 45° lat (1.0 x 1.0 degrees)
    bbox = [10.0, 45.0, 11.0, 46.0]
    with pytest.raises(ValueError, match="Area too large.*example"):
        validate_bbox_area(bbox, is_project=False, settings=Settings())


def test_validate_bbox_area_project_valid():
    """Should accept a batch/project area under 3000 km²."""
    # ~2,000 km² near 45° lat (0.5 x 0.5 degrees)
    bbox = [10.0, 45.0, 10.5, 45.5]
    area = validate_bbox_area(bbox, is_project=True, settings=Settings())
    assert 1500 < area < 2500


def test_validate_bbox_area_project_too_large():
    """Should reject project areas over 3000 km²."""
    # ~35,000 km² near 45° lat (2.0 x 2.0 degrees)
    bbox = [10.0, 45.0, 12.0, 47.0]
    with pytest.raises(ValueError, match="Area too large.*project"):
        validate_bbox_area(bbox, is_project=True, settings=Settings())
