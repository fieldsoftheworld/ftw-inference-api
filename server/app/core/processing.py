import asyncio
import json
import re
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .config import get_settings
from .geo import calculate_area_km2

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

        matcher = re.compile(r"^https?://[\w/.-]+$", re.A | re.I)
        for url in urls:
            if not matcher.match(url):
                raise ValueError(f"URL '{url}' contains invalid characters")
            result = urlparse(url)
            if not result.scheme or not result.netloc or not result.path:
                raise ValueError(f"URL '{url}' is invalid")

    # CHECK MODEL
    model_id = params.get("model")
    model_config = next(
        (model for model in settings.models if model.get("id") == model_id),
        None,
    )
    if not model_config:
        raise ValueError(f"Model with ID '{model_id}' not found")

    # Construct the full path to the model file
    model_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "models"
        / model_config.get("file")
    )
    if not model_path.exists():
        raise ValueError(f"Model file not found at '{model_path}'")
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
    uid = str(uuid.uuid4())
    image_file = TEMP_DIR / (uid + ".tif")
    inference_file = TEMP_DIR / (uid + ".inference.tif")
    polygon_file = TEMP_DIR / (uid + (".ndjson" if ndjson else ".json"))

    # Download and combine imagery
    # ftw inference download --out {output_path}
    #   --win_a {url_a} --win_b {url_b} --bbox {bbox}
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

    # ftw inference run {input} --out {output_path}
    #   --model {model} --resize_factor {resize}
    #   --patch_size {patch_size} --padding {padding}
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

    # ftw inference polygonize {input} --out {output_path}
    #   --simplify {simplify} --min_size {min_size} --close_interiors
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

    # Read the resulting GeoJSON and return it
    with open(polygon_file) as f:
        data = f.read() if ndjson else json.load(f)

    image_file.unlink(missing_ok=True)
    inference_file.unlink(missing_ok=True)
    polygon_file.unlink(missing_ok=True)

    return data


async def run_async(cmd):
    """Run subprocess command asynchronously"""
    # print(" ".join(cmd))
    # import time
    # start = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    # elapsed_time = time.perf_counter() - start
    # print(f"Elapsed time: {elapsed_time:.4f} seconds")

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

    return process
