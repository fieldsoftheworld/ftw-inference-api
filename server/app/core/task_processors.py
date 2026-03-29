import os
from functools import partial
from typing import Any

from app.benchmark.pipeline import run_benchmark_job
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.storage import StorageBackend
from app.core.types import ProjectStatus, TaskType
from app.services.inference_service import InferenceService
from app.services.project_service import ProjectService

logger = get_logger(__name__)


async def process_task(
    storage: StorageBackend, task_type: str, task_data: dict[str, Any]
) -> dict[str, Any]:
    """Generic task processor that handles common project management."""
    project_id = task_data["project_id"]

    project_service = ProjectService(storage)
    inference_service = InferenceService(storage, project_service)

    if task_type == TaskType.BENCHMARK.value:
        return await _handle_benchmark(task_data, inference_service)

    project_service.update_project_status(project_id, ProjectStatus.RUNNING)

    if task_type == TaskType.INFERENCE.value:
        result = await _handle_inference(task_data, inference_service)
    elif task_type == TaskType.POLYGONIZE.value:
        result = await _handle_polygonize(task_data, inference_service)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    if result.get("inference_file") or result.get("polygon_file"):
        task_type_enum = TaskType(task_type)
        project_service.record_task_completion(project_id, task_type_enum, result)

    return result


def _parse_bool_env(name: str) -> bool | None:
    """Return True/False if *name* is set in the environment, else None (use settings)."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


async def _handle_benchmark(
    task_data: dict[str, Any], inference_service: InferenceService
) -> dict[str, Any]:
    """Run FTW benchmark evaluation (no project DB row)."""
    from pathlib import Path

    from dotenv import load_dotenv

    from app.core.config import ENV_FILE_PATH, get_settings

    # Re-load .env and drop cached Settings so benchmark jobs always see repo .env
    # (nested benchmark.* from pydantic-settings can stay wrong across reloads otherwise).
    load_dotenv(ENV_FILE_PATH, override=True)
    get_settings.cache_clear()

    params = task_data["benchmark_params"]
    settings = get_settings()
    bm = settings.benchmark

    # Resolve data root: explicit cache_dir takes precedence over data_root.
    dr = getattr(bm, "cache_dir", None) or bm.data_root
    if isinstance(dr, str):
        dr = dr.strip() or None
    data_root = Path(dr).expanduser() if dr else None

    env_ad = _parse_bool_env("BENCHMARK__AUTO_DOWNLOAD")
    auto_download = bm.auto_download if env_ad is None else env_ad

    # JSON null sets the key to None; .get("k", True) returns None in that case, and
    # bool(None) is False — which disabled map_geojson. Treat None as "use default True".
    _imgj = params.get("include_map_geojson")
    include_map_geojson = True if _imgj is None else bool(_imgj)

    return await run_benchmark_job(
        inference_service,
        model_ids=params["model_ids"],
        country_ids=params["country_ids"],
        split=params.get("split", "test"),
        max_chips=params.get("max_chips", 10),
        seed=params.get("seed"),
        iou_threshold=params.get("iou_threshold", 0.25),
        data_root=data_root,
        allow_demo=bm.allow_demo,
        auto_download=auto_download,
        include_map_geojson=include_map_geojson,
    )


async def _handle_inference(
    task_data: dict, inference_service: InferenceService
) -> dict:
    """Handle inference specific processing."""
    project_id = task_data["project_id"]
    params = task_data["inference_params"]
    return await inference_service.run_project_inference(project_id, params)


async def _handle_polygonize(
    task_data: dict, inference_service: InferenceService
) -> dict:
    """Handle polygonization specific processing."""
    project_id = task_data["project_id"]
    params = task_data["polygon_params"]
    return await inference_service.run_project_polygonize(project_id, params)


def get_task_processors(storage: StorageBackend) -> dict[str, Any]:
    """Get dictionary of task type to processor function mappings."""
    base_processor = partial(process_task, storage)

    return {
        TaskType.INFERENCE.value: partial(base_processor, TaskType.INFERENCE.value),
        TaskType.POLYGONIZE.value: partial(base_processor, TaskType.POLYGONIZE.value),
        TaskType.BENCHMARK.value: partial(base_processor, TaskType.BENCHMARK.value),
    }
