"""ML pipeline orchestration functions."""

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

logger = get_logger(__name__)


async def download_images(
    image_file: Path,
    win_a: str,
    win_b: str,
    bbox: list[float],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Download satellite images using ftw tool."""
    download_start = time.time()
    logger.info("Downloading images", context)

    cmd = build_download_command(image_file, win_a, win_b, bbox)
    await run_async(cmd)

    download_time = round((time.time() - download_start) * 1000, 2)
    image_size_mb = round(image_file.stat().st_size / (1024 * 1024), 2)

    return {
        "download_time_ms": download_time,
        "image_size_mb": image_size_mb,
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
    win_b: str,
    inference_params: dict[str, Any],
    context: dict[str, Any] | None = None,
    gpu: int | None = None,
) -> dict[str, Any]:
    """Execute complete inference pipeline from download to inference."""
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
