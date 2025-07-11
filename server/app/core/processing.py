import asyncio
import json
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import get_settings
from .file_manager import get_project_file_manager
from .geo import calculate_area_km2
from .logging import get_logger

logger = get_logger(__name__)

TEMP_DIR = Path("data/temp")


def _build_download_command(
    image_file: Path, win_a: str, win_b: str, bbox: list[float]
) -> list[str]:
    """Build ftw inference download command."""
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


def _build_inference_command(
    image_file: Path, inference_file: Path, params: dict[str, Any]
) -> list[str]:
    """Build ftw inference run command."""
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

    # Add optional parameters
    if params.get("padding") is not None:
        cmd.extend(["--padding", str(params["padding"])])
    if params.get("patch_size") is not None:
        cmd.extend(["--patch_size", str(params["patch_size"])])

    return cmd


def _build_polygonize_command(
    inference_file: Path, polygon_file: Path, params: dict[str, Any]
) -> list[str]:
    """Build ftw inference polygonize command."""
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


async def _download_images(
    image_file: Path,
    win_a: str,
    win_b: str,
    bbox: list[float],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Download and combine imagery."""
    download_start = time.time()

    logger.info("Downloading images", context)

    cmd = _build_download_command(image_file, win_a, win_b, bbox)
    await run_async(cmd)

    download_time = round((time.time() - download_start) * 1000, 2)
    image_size_mb = round(image_file.stat().st_size / (1024 * 1024), 2)

    return {
        "download_time_ms": download_time,
        "image_size_mb": image_size_mb,
    }


async def _run_inference(
    image_file: Path,
    inference_file: Path,
    params: dict[str, Any],
    context: dict[str, Any] | None = None,
    gpu: int | None = None,
) -> dict[str, Any]:
    """Run ML inference on downloaded imagery."""
    inference_start = time.time()

    if context:
        logger.info("Starting ML inference", extra=context)

    cmd = _build_inference_command(image_file, inference_file, params)

    # Add GPU support if available
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    else:
        settings = get_settings()
        if settings.gpu is not None:
            cmd.extend(["--gpu", str(settings.gpu)])

    await run_async(cmd)

    inference_time = round((time.time() - inference_start) * 1000, 2)

    return {
        "inference_time_ms": inference_time,
    }


async def _run_polygonize(
    inference_file: Path,
    polygon_file: Path,
    params: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate polygons from inference results."""
    polygonize_start = time.time()

    if context:
        logger.info("Starting polygonization", extra=context)

    cmd = _build_polygonize_command(inference_file, polygon_file, params)
    await run_async(cmd)

    polygonize_time = round((time.time() - polygonize_start) * 1000, 2)

    return {
        "polygonize_time_ms": polygonize_time,
    }


async def _execute_inference_pipeline(
    image_file: Path,
    inference_file: Path,
    bbox: list[float],
    win_a: str,
    win_b: str,
    inference_params: dict[str, Any],
    context: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    gpu: int | None = None,
) -> dict[str, Any]:
    """Execute the complete inference pipeline."""
    start_time = time.time()

    try:
        # Download phase
        if progress_callback:
            progress_callback("Starting image download")

        download_result = await _download_images(
            image_file, win_a, win_b, bbox, context
        )

        # Inference phase
        if progress_callback:
            progress_callback("Running ML inference")

        inference_result = await _run_inference(
            image_file, inference_file, inference_params, context, gpu
        )

        # Combine results
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


def _validate_bbox(
    bbox: Any, require_bbox: bool = False, max_area: float | None = None
) -> None:
    """Validate bounding box parameters."""
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


def _validate_image_urls(urls: Any, require_image_urls: bool = False) -> None:
    """Validate image URL parameters."""
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


def _validate_model(params: dict[str, Any]) -> None:
    """Validate model parameters and resolve model path."""
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

    params["model"] = str(model_path.absolute())


def _validate_processing_params(params: dict[str, Any]) -> None:
    """Validate processing-specific parameters."""
    if params.get("resize_factor", 1) <= 0:
        raise ValueError("Resize factor must be a positive number")

    padding = params.get("padding")
    if padding is not None and padding < 0:
        raise ValueError("Padding must be null, a positive integer or 0")

    patch_size = params.get("patch_size")
    if patch_size is not None and patch_size % 32 != 0:
        raise ValueError("Patch size must be a multiple of 32.")


def prepare_inference_params(
    params, require_bbox=False, max_area=None, require_image_urls=False
):
    """Prepare and validate inference parameters."""
    _validate_bbox(params.get("bbox"), require_bbox, max_area)
    _validate_image_urls(params.get("images"), require_image_urls)
    _validate_model(params)
    _validate_processing_params(params)
    return params


def prepare_polygon_params(params):
    return params


async def run_example(inference_params, polygon_params, ndjson=False, gpu=None):
    """Run the example inference with the provided parameters."""
    uid = str(uuid.uuid4())
    image_file = TEMP_DIR / f"{uid}.tif"
    inference_file = TEMP_DIR / f"{uid}.inference.tif"
    polygon_file = TEMP_DIR / f"{uid}.{'ndjson' if ndjson else 'json'}"

    bbox = inference_params["bbox"]
    win_a = inference_params["images"][0]
    win_b = inference_params["images"][1]

    # Create logging context
    context = {
        "ml_metrics": {
            "processing_stage": "pipeline_start",
            "bounding_box": {
                "min_lon": bbox[0],
                "min_lat": bbox[1],
                "max_lon": bbox[2],
                "max_lat": bbox[3],
            },
            "bbox_area_km2": calculate_area_km2(bbox),
            "image_urls": [win_a, win_b],
            "model_path": inference_params.get("model", "unknown"),
            "gpu_enabled": gpu is not None,
        }
    }

    try:
        # Execute inference pipeline
        inference_result = await _execute_inference_pipeline(
            image_file,
            inference_file,
            bbox,
            win_a,
            win_b,
            inference_params,
            context,
            gpu=gpu,
        )

        # Execute polygonization
        polygon_result = await _run_polygonize(
            inference_file, polygon_file, polygon_params, context
        )

        # Read and return results
        with open(polygon_file) as f:
            data = f.read() if ndjson else json.load(f)

        # Count polygons
        if ndjson:
            polygons_generated = len(data.strip().split("\n")) if data.strip() else 0
        else:
            features = data.get("features", []) if isinstance(data, dict) else []
            polygons_generated = len(features)

        logger.info(
            "ML inference pipeline completed",
            extra={
                "ml_metrics": {
                    "processing_stage": "pipeline_complete",
                    "polygons_generated": polygons_generated,
                    "output_format": "ndjson" if ndjson else "geojson",
                    **inference_result,
                    **polygon_result,
                }
            },
        )

        return data

    finally:
        # Cleanup temporary files
        image_file.unlink(missing_ok=True)
        inference_file.unlink(missing_ok=True)
        polygon_file.unlink(missing_ok=True)


async def run_async(cmd):
    """Run subprocess command asynchronously"""
    # print(" ".join(cmd))
    # import time
    # start = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print(stdout.decode().strip())
    print(stderr.decode().strip())

    # elapsed_time = time.perf_counter() - start
    # print(f"Elapsed time: {elapsed_time:.4f} seconds")

    if process.returncode != 0:
        lines = stderr.decode().strip().splitlines()
        error_message = lines[-1]
        raise ValueError(error_message)

    return process


# Project-specific processing functions for async tasks

RESULTS_DIR = Path("data/results")


async def run_project_inference(
    project_id: str,
    params: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run inference for a project with hybrid URL/file support."""
    file_manager = get_project_file_manager(project_id)
    file_manager.ensure_directories()

    # Check if this is URL-based or file-based processing
    if params.get("images") and isinstance(params["images"], list):
        # URL-based processing (similar to /example endpoint)
        return await run_url_based_inference(project_id, params, progress_callback)
    else:
        # File-based processing using uploaded images
        return await run_file_based_inference(project_id, params, progress_callback)


async def run_url_based_inference(
    project_id: str,
    params: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Process inference using URLs (similar to /example endpoint)."""
    uid = str(uuid.uuid4())
    project_results_dir = RESULTS_DIR / project_id
    image_file = project_results_dir / f"{uid}.tif"
    inference_file = project_results_dir / f"{uid}.inference.tif"

    bbox = params["bbox"]
    win_a = params["images"][0]
    win_b = params["images"][1]

    # Create logging context
    context = {
        "ml_metrics": {
            "project_id": project_id,
            "processing_stage": "inference_start",
            "model_path": params["model"],
            "resize_factor": params["resize_factor"],
        }
    }

    try:
        # Execute inference pipeline
        result = await _execute_inference_pipeline(
            image_file,
            inference_file,
            bbox,
            win_a,
            win_b,
            params,
            context,
            progress_callback,
        )

        logger.info(
            "Project ML inference completed",
            extra={
                "ml_metrics": {
                    "processing_stage": "inference_complete",
                    "project_id": project_id,
                    **result,
                }
            },
        )

        return {
            "inference_file": str(inference_file),
            "image_file": str(image_file),
            **result,
        }

    except Exception as e:
        logger.error(
            "Project ML inference failed",
            exc_info=True,
            extra={
                "ml_context": {
                    "processing_stage": "inference_failed",
                    "project_id": project_id,
                    "error_type": type(e).__name__,
                }
            },
        )
        raise
    finally:
        # Only cleanup temporary image file, keep inference result
        image_file.unlink(missing_ok=True)


async def run_file_based_inference(
    project_id: str,
    params: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Process inference using uploaded files."""
    # TODO: Implement file-based processing for uploaded GeoTIFF files
    raise NotImplementedError("File-based processing not yet implemented")


async def run_project_polygonize(
    project_id: str,
    params: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run polygonization for a project."""
    start_time = time.time()
    file_manager = get_project_file_manager(project_id)

    inference_file = file_manager.get_latest_inference_result()
    if not inference_file:
        raise ValueError("No inference results found for this project")

    polygon_file = file_manager.get_result_path("polygons")

    if progress_callback:
        progress_callback("Starting polygonization")

    try:
        polygonize_start = time.time()
        logger.info(
            "Starting project polygonization",
            extra={
                "ml_metrics": {
                    "processing_stage": "polygonize_start",
                    "project_id": project_id,
                    "inference_file": str(inference_file),
                }
            },
        )

        polygonize_cmd = _build_polygonize_command(inference_file, polygon_file, params)
        await run_async(polygonize_cmd)

        polygonize_time = round((time.time() - polygonize_start) * 1000, 2)
        total_time = round((time.time() - start_time) * 1000, 2)

        # Read the resulting GeoJSON to count polygons
        with open(polygon_file) as f:
            geojson_data = json.load(f)

        features = (
            geojson_data.get("features", []) if isinstance(geojson_data, dict) else []
        )
        polygons_generated = len(features)

        logger.info(
            "Project polygonization completed",
            extra={
                "ml_metrics": {
                    "processing_stage": "polygonize_complete",
                    "project_id": project_id,
                    "polygonize_time_ms": polygonize_time,
                    "total_time_ms": total_time,
                    "polygons_generated": polygons_generated,
                }
            },
        )

        if progress_callback:
            progress_callback("Polygonization completed")

        return {
            "polygon_file": str(polygon_file),
            "polygonize_time_ms": polygonize_time,
            "total_time_ms": total_time,
            "polygons_generated": polygons_generated,
        }

    except Exception as e:
        logger.error(
            "Project polygonization failed",
            exc_info=True,
            extra={
                "ml_context": {
                    "processing_stage": "polygonize_failed",
                    "project_id": project_id,
                    "error_type": type(e).__name__,
                }
            },
        )
        raise
