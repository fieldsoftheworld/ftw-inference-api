import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.geo import calculate_area_km2
from app.core.logging import get_logger
from app.core.storage import StorageBackend, temp_files_context
from app.core.types import TaskType
from app.core.utils import run_async
from app.ml import (
    build_polygonize_command,
    execute_inference_pipeline,
    prepare_inference_params,
    run_polygonize,
)
from app.models.project import InferenceResult
from app.services.project_service import ProjectService
from app.services.task_service import TaskService

logger = get_logger(__name__)


class InferenceService:
    """Service for ML inference and polygonization operations."""

    def __init__(
        self,
        storage: StorageBackend,
        db: Session,
        project_service: ProjectService,
        task_service: TaskService | None = None,
    ):
        self.storage = storage
        self.db = db
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
                max_area=settings.max_area_km2,
            )
            polygon_params = params.get("polygons", {})
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e

        try:
            response_data = await self.run_example(
                inference_params, polygon_params, ndjson=ndjson, gpu=settings.gpu
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
            self.project_service.update_project_polygon_params(project_id, params)
            assert self.task_service is not None
            task_id = await self.task_service.submit_polygonize_task(project_id, params)
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
        inference_result = await self._get_latest_inference_result(project_id)
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

                with open(temp_polygon_file) as f:
                    geojson_data = json.load(f)

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

                return {
                    "polygon_file": polygon_key,
                    "polygon_key": polygon_key,
                    "polygonize_time_ms": polygonize_time,
                    "total_time_ms": total_time,
                    "polygons_generated": polygons_generated,
                }

            except Exception as e:
                self._log_ml_error("polygonization", e, project_id)
                raise

    async def run_example(
        self, inference_params, polygon_params, ndjson=False, gpu=None
    ):
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

            with open(polygon_file) as f:
                data = f.read() if ndjson else json.load(f)

            if ndjson:
                polygons_generated = (
                    len(data.strip().split("\n")) if data.strip() else 0
                )
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

                return {
                    "inference_file": inference_key,
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

    async def _get_latest_inference_result(
        self, project_id: str
    ) -> InferenceResult | None:
        """Get the most recent inference result for a project."""
        return (
            self.db.query(InferenceResult)
            .filter(
                InferenceResult.project_id == project_id,
                InferenceResult.result_type == "image",
            )
            .order_by(InferenceResult.created_at.desc())
            .first()
        )

    def _log_ml_start(self, stage: str, project_id: str | None = None, **extra) -> None:
        """Log ML pipeline start events with consistent structure."""
        context = {"processing_stage": f"{stage}_start", **extra}
        if project_id:
            context["project_id"] = project_id
        logger.info(f"Starting {stage}", extra={"ml_metrics": context})

    def _log_ml_success(
        self, stage: str, project_id: str | None = None, **extra
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
