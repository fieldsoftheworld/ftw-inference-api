import asyncio
import json
import logging
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .config import get_settings
from .geo import calculate_area_km2

logger = logging.getLogger(__name__)

TEMP_DIR = Path("data/temp")


def prepare_inference_params(
    params, require_bbox=False, max_area=None, require_image_urls=False
):
    settings = get_settings()

    # CHECK BBOX
    bbox = params.get("bbox")
    if require_bbox and not isinstance(bbox, list):
        raise ValueError("Bounding box is required as a list of four values.")
    elif isinstance(bbox, list):
        # Ensure bbox is in the correct format
        if len(bbox) != 4:
            raise ValueError("Bounding box must be in format [minX, minY, maxX, maxY]")

        # Verify that bbox is within EPSG:4326 bounds
        min_lon, min_lat, max_lon, max_lat = bbox

        if min_lon < -180 or max_lon > 180:
            raise ValueError(
                "Longitude values must be between -180 and 180 degrees in EPSG:4326"
            )

        if min_lat < -90 or max_lat > 90:
            raise ValueError(
                "Latitude values must be between -90 and 90 degrees in EPSG:4326"
            )

        # Ensure min < max for both dimensions
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(
                "Invalid bounding box: min values must be less than max values"
            )

        if max_area is not None:
            area_size_km2 = calculate_area_km2(bbox)
            if area_size_km2 > max_area:
                raise ValueError(
                    "Area too large for example endpoint. "
                    + f"Maximum allowed area is {max_area} km². "
                    + "Please create a project instead.",
                )

    # CHECK IMAGES
    urls = params.get("images")
    if require_image_urls:
        if not isinstance(urls, list) or len(urls) != 2:
            raise ValueError("Images must be a list of two items")

        for url in urls:
            result = urlparse(url)
            if not result.scheme or not result.netloc or not result.path:
                raise ValueError(f"URL {url} is invalid")

    # CHECK MODEL
    model_id = params.get("model")
    model_config = next(
        (model for model in settings.models if model.get("id") == model_id),
        None,
    )
    if not model_config:
        raise ValueError(f"Model with ID {model_id} not found")

    # Construct the full path to the model file
    model_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "models"
        / model_config.get("file")
    )
    if not model_path.exists():
        raise ValueError(f"Model file not found at {model_path}")
    else:
        params["model"] = str(model_path.absolute())

    # CHECK RESIZE FACTOR
    if params.get("resize_factor") <= 0:
        raise ValueError("Resize factor must be a positive number")

    # CHECK PADDING
    if params.get("padding") < 0:
        raise ValueError("Padding must be a positive integer or 0")

    # CHECK PATCH SIZE
    patch_size = params.get("patch_size")
    if patch_size is not None and patch_size % 32 != 0:
        raise ValueError("Patch size must be a multiple of 32.")

    return params


def prepare_polygon_params(params):
    return params


async def run_example(inference_params, polygon_params, ndjson=False, gpu=None):
    """
    Run the example inference with the provided parameters.
    """
    import time
    
    uid = str(uuid.uuid4())
    image_file = TEMP_DIR / (uid + ".tif")
    inference_file = TEMP_DIR / (uid + ".inference.tif")
    polygon_file = TEMP_DIR / (uid + ".ndjson" if ndjson else ".json")

    try:
        # Download and combine imagery
        start_time = time.perf_counter()
        win_a = inference_params["images"][0]
        win_b = inference_params["images"][1]
        bbox = ",".join(map(str, inference_params["bbox"]))
        download_cmd = [
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
            bbox,
        ]
        await run_async(download_cmd)
        download_time = time.perf_counter() - start_time
        logger.info(f"Download completed in {download_time:.2f}s")

        # Run ML inference
        start_time = time.perf_counter()
        inference_cmd = [
            "ftw",
            "inference",
            "run",
            str(image_file.absolute()),
            "--overwrite",
            "--out",
            str(inference_file.absolute()),
            "--model",
            inference_params["model"],
            "--resize_factor",
            str(inference_params["resize_factor"]),
            "--padding",
            str(inference_params["padding"]),
        ]
        patch_size = inference_params.get("patch_size")
        if patch_size is not None:
            inference_cmd.extend(["--patch_size", str(patch_size)])
        if gpu is not None:
            inference_cmd.extend(["--gpu", str(gpu)])

        await run_async(inference_cmd)
        inference_time = time.perf_counter() - start_time
        logger.info(f"ML inference completed in {inference_time:.2f}s")

        # Run polygonization
        start_time = time.perf_counter()
        polygonize_cmd = [
            "ftw",
            "inference",
            "polygonize",
            str(inference_file.absolute()),
            "--overwrite",
            "--out",
            str(polygon_file.absolute()),
            "--simplify",
            str(polygon_params["simplify"]),
            "--min_size",
            str(polygon_params["min_size"]),
        ]
        if polygon_params["close_interiors"]:
            polygonize_cmd.append("--close_interiors")

        await run_async(polygonize_cmd)
        polygonize_time = time.perf_counter() - start_time
        logger.info(f"Polygonization completed in {polygonize_time:.2f}s")

        # Read the resulting GeoJSON and return it
        with open(polygon_file) as f:
            data = f.read() if ndjson else json.load(f)

        # Cleanup temp files
        image_file.unlink(missing_ok=True)
        inference_file.unlink(missing_ok=True)
        polygon_file.unlink(missing_ok=True)

        total_time = download_time + inference_time + polygonize_time
        logger.info(f"Pipeline completed - Total: {total_time:.2f}s (Download: {download_time:.2f}s, Inference: {inference_time:.2f}s, Polygonize: {polygonize_time:.2f}s)")

        return data

    except Exception as e:
        # Log error with context for debugging concurrent failures
        logger.error(f"Pipeline failed for request {uid}: {str(e)}")
        logger.error(f"Request details - Model: {inference_params.get('model', 'unknown')}, "
                    f"Bbox: {inference_params.get('bbox', 'unknown')}, "
                    f"Images: {len(inference_params.get('images', []))} provided")
        
        # Cleanup on failure
        image_file.unlink(missing_ok=True)
        inference_file.unlink(missing_ok=True)
        polygon_file.unlink(missing_ok=True)
        
        raise


async def run_async(cmd):
    """Run subprocess command asynchronously"""
    import os
    
    # If command starts with 'ftw', prepend with 'uv run'
    if cmd[0] == "ftw":
        cmd = ["uv", "run"] + cmd
    
    # Set CUDA environment variables for GPU support
    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            # Log detailed error information for debugging
            cmd_str = ' '.join(cmd)
            stderr_str = stderr.decode().strip() if stderr else "No stderr"
            stdout_str = stdout.decode().strip() if stdout else "No stdout"
            
            logger.error(f"Command failed: {cmd_str}")
            logger.error(f"Return code: {process.returncode}")
            logger.error(f"STDERR: {stderr_str}")
            logger.error(f"STDOUT: {stdout_str}")
            
            # Create a more informative error message
            error_msg = f"Command failed with return code {process.returncode}"
            if stderr_str and stderr_str != "No stderr":
                error_msg += f": {stderr_str}"
            
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

        return process
        
    except Exception as e:
        # Log any other exceptions (memory issues, process creation failures, etc.)
        cmd_str = ' '.join(cmd)
        logger.error(f"Failed to execute command: {cmd_str}")
        logger.error(f"Exception: {str(e)}")
        raise
