"""CLI command builders for ftw tool."""

from pathlib import Path
from typing import Any

from app.ml.validation import resolve_model_path


def build_download_command(
    image_file: Path, win_a: str, win_b: str, bbox: list[float]
) -> list[str]:
    """Build ftw download command for image acquisition."""
    bbox_str = ",".join(map(str, bbox))
    return [
        "ftw",
        "inference",
        "download",
        "--out",
        str(image_file.absolute()),
        "--win_a",
        win_a,
        "--win_b",
        win_b,
        "--bbox",
        bbox_str,
    ]


def build_inference_command(
    image_file: Path, inference_file: Path, params: dict[str, Any]
) -> list[str]:
    """Build ftw inference command for ML processing."""
    cmd = [
        "ftw",
        "inference",
        "run",
        str(image_file.absolute()),
        "--overwrite",
        "--out",
        str(inference_file.absolute()),
        "--model",
        resolve_model_path(params["model"]),
        "--resize_factor",
        str(params["resize_factor"]),
    ]

    if params.get("padding") is not None:
        cmd.extend(["--padding", str(params["padding"])])
    if params.get("patch_size") is not None:
        cmd.extend(["--patch_size", str(params["patch_size"])])

    return cmd


def build_polygonize_command(
    inference_file: Path, polygon_file: Path, params: dict[str, Any]
) -> list[str]:
    """Build ftw polygonize command for vector conversion."""
    cmd = [
        "ftw",
        "inference",
        "polygonize",
        str(inference_file.absolute()),
        "--overwrite",
        "--out",
        str(polygon_file.absolute()),
        "--simplify",
        str(params["simplify"]),
        "--min_size",
        str(params["min_size"]),
    ]

    if params.get("close_interiors", False):
        cmd.append("--close_interiors")

    return cmd


def build_scene_selection_command(
    year: int,
    bbox: list[float],
    out: str,
    cloud_cover_max: int = 20,
    buffer_days: int = 14,
) -> list[str]:
    """Build ftw scene_selection command for Sentinel-2 scene selection."""
    bbox_str = ",".join(map(str, bbox))
    cmd = [
        "ftw",
        "inference",
        "scene_selection",
        "--year",
        str(year),
        "--bbox",
        bbox_str,
        "--cloud_cover_max",
        str(cloud_cover_max),
        "--buffer_days",
        str(buffer_days),
        "--out",
        out,
    ]

    return cmd
