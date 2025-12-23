"""CLI command builders for ftw tool."""

from pathlib import Path
from typing import Any


def build_download_command(
    image_file: Path, win_a: str, win_b: str | None, bbox: list[float]
) -> list[str]:
    """Build ftw download command. Handles both single and dual window."""
    bbox_str = ",".join(map(str, bbox))
    cmd = [
        "ftw",
        "inference",
        "download",
        "--out",
        str(image_file.absolute()),
        "--win_a",
        win_a,
        "--bbox",
        bbox_str,
    ]

    # Only add win_b if provided (for dual-window models)
    if win_b is not None:
        cmd.extend(["--win_b", win_b])

    return cmd


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
        params["model"],
        "--resize_factor",
        str(params["resize_factor"]),
    ]

    if params.get("padding") is not None:
        cmd.extend(["--padding", str(params["padding"])])
    if params.get("patch_size") is not None:
        cmd.extend(["--patch_size", str(params["patch_size"])])

    return cmd


def build_instance_segmentation_command(
    image_file: Path,
    output_file: Path,
    params: dict[str, Any],
) -> list[str]:
    """Build ftw instance segmentation command (outputs GeoJSON directly)."""
    cmd = [
        "ftw",
        "inference",
        "run-instance-segmentation",
        str(image_file.absolute()),
        "--overwrite",
        "--out",
        str(output_file.absolute()),
        "--model",
        params["model"],
    ]

    if params.get("resize_factor") is not None:
        cmd.extend(["--resize_factor", str(params["resize_factor"])])

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
    stac_host: str = "mspc",
    s2_collection: str = "c1",
) -> list[str]:
    """Build ftw scene-selection command for Sentinel-2 scene selection."""
    bbox_str = ",".join(map(str, bbox))
    cmd = [
        "ftw",
        "inference",
        "scene-selection",
        "--year",
        str(year),
        "--bbox",
        bbox_str,
        "--cloud_cover_max",
        str(cloud_cover_max),
        "--buffer_days",
        str(buffer_days),
        "--stac_host",
        stac_host,
        "--s2_collection",
        s2_collection,
        "--out",
        out,
    ]

    return cmd
