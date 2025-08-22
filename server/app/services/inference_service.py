import json
import time
import uuid
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.geo import calculate_area_km2
from app.core.logging import get_logger
from app.core.storage import StorageBackend, temp_files_context
from app.core.types import TaskType
from app.core.utils import run_async
from app.db.models import InferenceResult
from app.ml import (
    build_polygonize_command,
    execute_inference_pipeline,
    prepare_inference_params,
    run_polygonize,
)
from app.ml.commands import build_scene_selection_command
from app.ml.validation import prepare_scene_selection_params
from app.services.project_service import ProjectService
from app.services.task_service import TaskService

logger = get_logger(__name__)


class InferenceService:
    """Service for ML inference and polygonization operations."""

    def __init__(
        self,
        storage: StorageBackend,
        project_service: ProjectService,
        task_service: TaskService | None = None,
    ):
        self.storage = storage
        self.project_service = project_service
        self.task_service = task_service

    # --- Static Methods ---

    @staticmethod
    def prepare_inference_params(
        params: dict[str, Any],
        require_bbox: bool = False,
        max_area: float | None = None,
        require_image_urls: bool = False,
    ) -> dict[str, Any]:
        """Prepare and validate inference parameters using ML module."""
        return prepare_inference_params(
            params, require_bbox, max_area, require_image_urls
        )

    # --- Public API: Workflow Submission ---

    async def run_scene_selection(self, params: dict[str, Any]) -> dict[str, str]:
        """Run scene selection to find optimal Sentinel-2 scenes."""
        try:
            # Validate parameters
            validated_params = prepare_scene_selection_params(params)

            async with temp_files_context() as temp_files:
                # Create temporary output file
                temp_output = temp_files.create_temp_file(suffix=".json")

                # Build and execute CLI command
                cmd = build_scene_selection_command(
                    year=validated_params["year"],
                    bbox=validated_params["bbox"],
                    out=str(temp_output),
                    cloud_cover_max=validated_params.get("cloud_cover_max", 20),
                    buffer_days=validated_params.get("buffer_days", 14),
                )

                result = await run_async(cmd)
                if result.returncode != 0:
                    logger.error(f"Scene selection failed: {result.stderr}")
                    raise RuntimeError("Scene selection command failed")

                # Read and parse JSON output
                async with aiofiles.open(temp_output) as f:
                    content = await f.read()
                    scene_data = json.loads(content)

                # Convert S3 URLs to HTTP URLs
                def s3_to_http(s3_url: str) -> str:
                    """Convert s3://bucket/path to https://bucket.s3.amazonaws.com/path"""
                    if s3_url.startswith("s3://"):
                        # Remove s3:// prefix and split bucket from path
                        s3_path = s3_url[5:]  # Remove "s3://"
                        parts = s3_path.split("/", 1)
                        if len(parts) == 2:
                            bucket, path = parts
                            return f"https://{bucket}.s3.amazonaws.com/{path}"
                    return s3_url  # Return as-is if not S3 format

                # Convert both windows to HTTP URLs
                return {
                    "window_a": s3_to_http(scene_data.get("window_a", "")),
                    "window_b": s3_to_http(scene_data.get("window_b", "")),
                }

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e
        except Exception as e:
            logger.error(f"Scene selection error: {e!s}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during scene selection",
            ) from e

    async def run_example_workflow(
        self,
        params: dict[str, Any],
        accept_header: str | None = None,
    ) -> dict[str, Any]:
        """Run complete example workflow with logging and format handling."""
        settings = get_settings()
        ndjson = accept_header and "application/x-ndjson" in accept_header

        if not params.get("inference"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inference parameters are required",
            )

        try:
            inference_params = self.prepare_inference_params(
                params["inference"],
                require_bbox=True,
                require_image_urls=True,
                max_area=settings.processing.max_area_km2,
            )
            polygon_params = params.get("polygons", {})
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e

        try:
            response_data = await self.run_example(
                inference_params,
                polygon_params,
                ndjson=bool(ndjson) if ndjson is not None else False,
                gpu=settings.processing.gpu,
            )

            return {
                "data": response_data,
                "format": "ndjson" if ndjson else "geojson",
                "media_type": "application/x-ndjson"
                if ndjson
                else "application/geo+json",
            }

        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            self._log_ml_error("example_workflow", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while processing the request",
            ) from e

    async def submit_project_inference_workflow(
        self,
        project_id: str,
        params: dict[str, Any],
    ) -> str:
        """Submit inference workflow for a project and return task_id."""
        try:
            inference_params = self.prepare_inference_params(params)
            self.project_service.update_project_inference_params(
                project_id, inference_params
            )
            assert self.task_service is not None
            task_id = await self.task_service.submit_inference_task(
                project_id, inference_params
            )
            self.project_service.set_project_task_id(
                project_id, task_id, TaskType.INFERENCE
            )
            return task_id

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e
        except Exception as e:
            self._log_ml_error("workflow_submission", e, project_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during task submission.",
            ) from e

    async def submit_project_polygonize_workflow(
        self,
        project_id: str,
        params: dict[str, Any],
    ) -> str:
        """Submit polygonize workflow for a project and return task_id."""
        try:
            poly_params = params
            self.project_service.update_project_polygon_params(project_id, poly_params)
            assert self.task_service is not None
            task_id = await self.task_service.submit_polygonize_task(
                project_id, poly_params
            )
            self.project_service.set_project_task_id(
                project_id, task_id, TaskType.POLYGONIZE
            )
            return task_id

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e
        except Exception as e:
            self._log_ml_error("workflow_submission", e, project_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during task submission.",
            ) from e

    # --- Public API: Core ML Execution ---

    async def run_project_inference(
        self,
        project_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run ML inference for a project."""
        if params.get("images") and isinstance(params["images"], list):
            return await self._run_url_based_inference(project_id, params)
        else:
            return await self._run_file_based_inference(project_id, params)

    async def run_project_polygonize(
        self,
        project_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run polygonization for a project."""
        start_time = time.time()
        uid = str(uuid.uuid4())

        # Get the latest inference result for this project
        inference_result = self._get_latest_inference_result(project_id)
        if not inference_result:
            raise ValueError("No inference results found for this project")

        polygon_result_key = f"projects/{project_id}/results/polygons_{uid}.json"

        async with temp_files_context(
            f"{uid}.inference.tif", f"{uid}.polygons.json"
        ) as (temp_inference_file, temp_polygon_file):
            try:
                polygonize_start = time.time()

                # Download inference file from storage
                await self.storage.download(
                    inference_result.file_path, temp_inference_file
                )

                self._log_ml_start(
                    "polygonization",
                    project_id,
                    inference_file=str(temp_inference_file),
                )

                polygonize_cmd = build_polygonize_command(
                    temp_inference_file, temp_polygon_file, params
                )
                await run_async(polygonize_cmd)

                polygonize_time = round((time.time() - polygonize_start) * 1000, 2)
                total_time = round((time.time() - start_time) * 1000, 2)

                async with aiofiles.open(temp_polygon_file) as f:
                    geojson_content = await f.read()
                    geojson_data = json.loads(geojson_content)

                features = (
                    geojson_data.get("features", [])
                    if isinstance(geojson_data, dict)
                    else []
                )
                polygons_generated = len(features)

                polygon_key = await self.storage.upload(
                    temp_polygon_file, polygon_result_key
                )

                self._log_ml_success(
                    "polygonization",
                    project_id,
                    polygonize_time_ms=polygonize_time,
                    total_time_ms=total_time,
                    polygons_generated=polygons_generated,
                    s3_key=polygon_key,
                )

                polygon_file_url = await self.storage.get_url(polygon_key)

                return {
                    "polygon_file": polygon_file_url,
                    "polygon_key": polygon_key,
                    "polygonize_time_ms": polygonize_time,
                    "total_time_ms": total_time,
                    "polygons_generated": polygons_generated,
                }

            except Exception as e:
                self._log_ml_error("polygonization", e, project_id)
                raise

    async def run_example(
        self,
        inference_params: dict[str, Any],
        polygon_params: dict[str, Any],
        ndjson: bool = False,
        gpu: int | None = None,
    ) -> str | dict[str, Any]:
        """Run complete ML pipeline for example workflow."""
        uid = str(uuid.uuid4())
        temp_dir = Path("data/temp")
        image_file = temp_dir / f"{uid}.tif"
        inference_file = temp_dir / f"{uid}.inference.tif"
        polygon_file = temp_dir / f"{uid}.{'ndjson' if ndjson else 'json'}"

        bbox = inference_params["bbox"]
        win_a = inference_params["images"][0]
        win_b = inference_params["images"][1]

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
            inference_result = await execute_inference_pipeline(
                image_file,
                inference_file,
                bbox,
                win_a,
                win_b,
                inference_params,
                context,
                gpu=gpu,
            )

            polygon_result = await run_polygonize(
                inference_file, polygon_file, polygon_params, context
            )

            async with aiofiles.open(polygon_file) as f:
                if ndjson:
                    data: str | dict[str, Any] = await f.read()
                else:
                    content = await f.read()
                    data = json.loads(content)

            if ndjson:
                if isinstance(data, str):
                    polygons_generated = (
                        len(data.strip().split("\n")) if data.strip() else 0
                    )
                else:
                    polygons_generated = 0
            else:
                features = data.get("features", []) if isinstance(data, dict) else []
                polygons_generated = len(features)

            self._log_ml_success(
                "pipeline",
                polygons_generated=polygons_generated,
                output_format="ndjson" if ndjson else "geojson",
                **inference_result,
                **polygon_result,
            )

            return data

        finally:
            image_file.unlink(missing_ok=True)
            inference_file.unlink(missing_ok=True)
            polygon_file.unlink(missing_ok=True)

    # --- Internal Helper Methods ---

    async def _run_url_based_inference(
        self,
        project_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run inference using URL-based image processing."""
        uid = str(uuid.uuid4())
        inference_result_key = f"projects/{project_id}/results/inference_{uid}.tif"

        bbox = params["bbox"]
        win_a = params["images"][0]
        win_b = params["images"][1]

        context = {
            "ml_metrics": {
                "project_id": project_id,
                "processing_stage": "inference_start",
                "model_path": params["model"],
                "resize_factor": params["resize_factor"],
            }
        }

        async with temp_files_context(f"{uid}.tif", f"{uid}.inference.tif") as (
            temp_image_file,
            temp_inference_file,
        ):
            try:
                result = await execute_inference_pipeline(
                    temp_image_file,
                    temp_inference_file,
                    bbox,
                    win_a,
                    win_b,
                    params,
                    context,
                )

                inference_key = await self.storage.upload(
                    temp_inference_file, inference_result_key
                )

                self._log_ml_success(
                    "inference",
                    project_id,
                    s3_key=inference_key,
                    **result,
                )

                inference_file_url = await self.storage.get_url(inference_key)

                return {
                    "inference_file": inference_file_url,
                    "inference_key": inference_key,
                    "image_file": str(temp_image_file),
                    **result,
                }

            except Exception as e:
                self._log_ml_error("inference", e, project_id)
                raise

    async def _run_file_based_inference(
        self,
        project_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run inference using file-based image processing."""
        raise NotImplementedError("File-based processing not yet implemented")

    def _get_latest_inference_result(self, project_id: str) -> InferenceResult | None:
        """Get the most recent inference result for a project."""
        result: InferenceResult | None = InferenceResult.get_latest_by_project_and_type(
            project_id, "image"
        )
        return result

    def _log_ml_start(
        self, stage: str, project_id: str | None = None, **extra: Any
    ) -> None:
        """Log ML pipeline start events with consistent structure."""
        context = {"processing_stage": f"{stage}_start", **extra}
        if project_id:
            context["project_id"] = project_id
        logger.info(f"Starting {stage}", extra={"ml_metrics": context})

    def _log_ml_success(
        self, stage: str, project_id: str | None = None, **extra: Any
    ) -> None:
        """Log ML pipeline success events with consistent structure."""
        context = {"processing_stage": f"{stage}_complete", **extra}
        if project_id:
            context["project_id"] = project_id
        logger.info(f"{stage.title()} completed", extra={"ml_metrics": context})

    def _log_ml_error(
        self, stage: str, error: Exception, project_id: str | None = None
    ) -> None:
        """Log ML pipeline errors with consistent structure."""
        context = {
            "processing_stage": f"{stage}_failed",
            "error_type": type(error).__name__,
        }
        if project_id:
            context["project_id"] = project_id
        logger.error(
            f"{stage.title()} failed", exc_info=True, extra={"ml_metrics": context}
        )
