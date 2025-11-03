"""ML pipeline orchestration functions."""

import shutil
import time
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.utils import run_async
from app.ml.commands import (
    build_download_command,
    build_inference_command,
    build_polygonize_command,
)
from app.services.cache_service import check_s2_scene_exists, save_to_cache

logger = get_logger(__name__)


async def download_images(
    image_file: Path,
    win_a: str,
    win_b: str | None,
    bbox: list[float],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Download satellite images using ftw tool. Handles both single and dual window."""
    download_start = time.time()

    # Check cache first (only for dual window scenarios)
    if win_b is not None:
        cached, cached_path = await check_s2_scene_exists(win_a, win_b, bbox)

        if cached and cached_path is not None:
            logger.info("Using cached S2 scene", extra=context)
            shutil.copy2(cached_path, image_file)
            download_time = round((time.time() - download_start) * 1000, 2)
            image_size_mb = round(image_file.stat().st_size / (1024 * 1024), 2)

            return {
                "download_time_ms": download_time,
                "image_size_mb": image_size_mb,
                "from_cache": True,
            }

    # Not cached or single window - download from S3
    logger.info("Downloading images from S3", extra=context)
    cmd = build_download_command(image_file, win_a, win_b, bbox)
    await run_async(cmd)

    download_time = round((time.time() - download_start) * 1000, 2)
    image_size_mb = round(image_file.stat().st_size / (1024 * 1024), 2)

    # Save to cache for next time (only for dual window scenarios)
    if win_b is not None:
        await save_to_cache(image_file, win_a, win_b, bbox)

    return {
        "download_time_ms": download_time,
        "image_size_mb": image_size_mb,
        "from_cache": False,
    }


async def run_inference(
    image_file: Path,
    inference_file: Path,
    params: dict[str, Any],
    context: dict[str, Any] | None = None,
    gpu: int | None = None,
) -> dict[str, Any]:
    """Run ML inference on downloaded images."""
    settings = get_settings()
    inference_start = time.time()

    if context:
        logger.info("Starting ML inference", extra=context)

    cmd = build_inference_command(image_file, inference_file, params)

    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    elif settings.processing.gpu is not None:
        cmd.extend(["--gpu", str(settings.processing.gpu)])

    await run_async(cmd)

    inference_time = round((time.time() - inference_start) * 1000, 2)

    return {
        "inference_time_ms": inference_time,
    }


async def run_polygonize(
    inference_file: Path,
    polygon_file: Path,
    params: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert inference results to polygon vectors."""
    polygonize_start = time.time()

    if context:
        logger.info("Starting polygonization", extra=context)

    cmd = build_polygonize_command(inference_file, polygon_file, params)
    await run_async(cmd)

    polygonize_time = round((time.time() - polygonize_start) * 1000, 2)

    return {
        "polygonize_time_ms": polygonize_time,
    }


async def execute_inference_pipeline(
    image_file: Path,
    inference_file: Path,
    bbox: list[float],
    win_a: str,
    win_b: str | None,
    inference_params: dict[str, Any],
    context: dict[str, Any] | None = None,
    gpu: int | None = None,
) -> dict[str, Any]:
    """Execute ML inference pipeline with single/dual window support."""
    start_time = time.time()

    try:
        download_result = await download_images(image_file, win_a, win_b, bbox, context)

        inference_result = await run_inference(
            image_file, inference_file, inference_params, context, gpu
        )

        total_time = round((time.time() - start_time) * 1000, 2)

        return {
            **download_result,
            **inference_result,
            "total_time_ms": total_time,
        }

    except Exception as e:
        total_time = round((time.time() - start_time) * 1000, 2)
        if context:
            logger.error(
                "Inference pipeline failed",
                exc_info=True,
                extra={
                    "ml_context": {
                        "processing_stage": "pipeline_failed",
                        "total_time_ms": total_time,
                        "error_type": type(e).__name__,
                        **context.get("ml_metrics", {}),
                    }
                },
            )
        raise
