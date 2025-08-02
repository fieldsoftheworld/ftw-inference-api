import json
import uuid
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.storage import StorageBackend, temp_files_context
from app.core.types import ProjectStatus, TaskType
from app.models.project import Image, InferenceResult, Project
from app.schemas import CreateProjectRequest, ProjectResponse, ProjectResultLinks
from app.services.task_service import TaskService

logger = get_logger(__name__)


def _clean_parameters_for_response(parameters: Any) -> dict[str, Any]:
    """Clean parameters for API response, excluding large fields."""
    params_dict = _normalize_parameters(parameters)
    if not params_dict:
        return {}

    clean_params: dict[str, Any] = {}
    _process_inference_params(params_dict, clean_params)
    _copy_direct_params(params_dict, clean_params)
    return clean_params


def _normalize_parameters(parameters: Any) -> dict[str, Any]:
    """Convert parameters to dictionary format."""
    if isinstance(parameters, dict):
        return parameters
    result: dict[str, Any] = (
        parameters.model_dump() if hasattr(parameters, "model_dump") else {}
    )
    return result


def _process_inference_params(
    params_dict: dict[str, Any], clean_params: dict[str, Any]
) -> None:
    """Process inference-specific parameters."""
    inf_params = params_dict.get("inference")
    if not inf_params:
        return

    clean_params["inference"] = {k: v for k, v in inf_params.items() if k != "images"}

    _process_model_field(inf_params, clean_params["inference"])
    _process_images_count(inf_params, clean_params["inference"])


def _process_model_field(
    inf_params: dict[str, Any], clean_inference: dict[str, Any]
) -> None:
    """Process model field, cleaning checkpoint paths."""
    model_value = inf_params.get("model")
    if not model_value:
        return

    if isinstance(model_value, str) and model_value.endswith(".ckpt"):
        clean_inference["model"] = model_value.split("/")[-1].replace(".ckpt", "")
    else:
        clean_inference["model"] = model_value


def _process_images_count(
    inf_params: dict[str, Any], clean_inference: dict[str, Any]
) -> None:
    """Add images count instead of full images data."""
    images = inf_params.get("images")
    if images:
        clean_inference["images_count"] = len(images)


def _copy_direct_params(
    params_dict: dict[str, Any], clean_params: dict[str, Any]
) -> None:
    """Copy parameters that don't need processing."""
    for key in ["polygons", "task_id", "polygonize_task_id"]:
        if key in params_dict:
            clean_params[key] = params_dict[key]


class ProjectService:
    def __init__(self, storage: StorageBackend, db: Session):
        """Initialize ProjectService with storage backend and database session."""
        self.storage = storage
        self.db = db

    # --- Public API: Project Lifecycle & Management ---

    def create_project(self, project_data: CreateProjectRequest) -> Project:
        """Create a new project."""
        new_project = Project(title=project_data.title)
        self.db.add(new_project)
        self.db.commit()
        self.db.refresh(new_project)
        return new_project

    async def get_project(self, project_id: str) -> ProjectResponse:
        """Get a single project by ID with clean URLs."""
        project = self._get_project_or_404(project_id)
        clean_results = await self._get_project_results_urls(project)

        return ProjectResponse(
            id=project.id,
            title=project.title,
            status=project.status,
            progress=project.progress,
            created_at=project.created_at,
            parameters=_clean_parameters_for_response(project.parameters),
            results=clean_results,
        )

    async def get_projects(self) -> list[ProjectResponse]:
        """Get all projects with clean URLs."""
        projects = self.db.query(Project).all()
        clean_projects = []

        for project in projects:
            clean_results = await self._get_project_results_urls(project)

            clean_project = ProjectResponse(
                id=project.id,
                title=project.title,
                status=project.status,
                progress=project.progress,
                created_at=project.created_at,
                parameters=_clean_parameters_for_response(project.parameters),
                results=clean_results,
            )
            clean_projects.append(clean_project)

        return clean_projects

    async def delete_project(self, project_id: str) -> None:
        """Delete a project and all its associated storage files."""
        project = self._get_project_or_404(project_id)

        await self._cleanup_project_files(project_id)

        self.db.delete(project)
        self.db.commit()

    def get_project_status(self, project_id: str) -> dict[str, Any]:
        """Get basic project status information."""
        project = self._get_project_or_404(project_id)

        return {
            "project_id": project_id,
            "status": project.status,
            "progress": project.progress,
            "parameters": _clean_parameters_for_response(project.parameters),
        }

    async def get_complete_project_status(
        self, project_id: str, task_service: TaskService
    ) -> dict[str, Any]:
        """Get complete project status with aggregated task info."""
        response_data = self.get_project_status(project_id)

        if "task_id" in response_data["parameters"]:
            task_info = await task_service.get_task_info(
                response_data["parameters"]["task_id"]
            )
            if task_info:
                response_data["task"] = task_info

        if "polygonize_task_id" in response_data["parameters"]:
            poly_task_info = await task_service.get_task_info(
                response_data["parameters"]["polygonize_task_id"]
            )
            if poly_task_info:
                response_data["polygonize_task"] = poly_task_info

        return response_data

    def update_project_status(self, project_id: str, new_status: ProjectStatus) -> None:
        """Update project status."""
        project = self._get_project_or_404(project_id)
        project.status = new_status
        self.db.commit()

    # --- Public API: File & Parameter Management ---

    async def upload_image(
        self, project_id: str, window: str, file: UploadFile
    ) -> None:
        """Upload an image file for a project window (a or b)."""
        if window not in ["a", "b"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Window must be 'a' or 'b'",
            )

        self._get_project_or_404(project_id)

        if not self.storage:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Storage backend not configured",
            )

        async with aiofiles.tempfile.NamedTemporaryFile(
            delete=False, suffix=".tif"
        ) as temp_file:
            temp_path = Path(str(temp_file.name))
            content = await file.read()
            await temp_file.write(content)

        try:
            s3_key = f"projects/{project_id}/uploads/{window}/{uuid.uuid4()}.tif"
            await self.storage.upload(temp_path, s3_key)
        finally:
            temp_path.unlink(missing_ok=True)

        existing_image = (
            self.db.query(Image)
            .filter(Image.project_id == project_id, Image.window == window)
            .first()
        )

        if existing_image:
            existing_image.file_path = s3_key
            self.db.commit()
            self.db.refresh(existing_image)
        else:
            new_image = Image(project_id=project_id, window=window, file_path=s3_key)
            self.db.add(new_image)
            self.db.commit()

    def update_project_inference_params(
        self, project_id: str, inference_params: dict[str, Any]
    ) -> None:
        """Update inference parameters for a project and reset status to queued."""
        self._update_project_params(project_id, "inference", inference_params)

    def update_project_polygon_params(
        self, project_id: str, polygon_params: dict[str, Any]
    ) -> None:
        """Update polygon parameters for a project and reset status to queued."""
        self._update_project_params(project_id, "polygons", polygon_params)

    def set_project_task_id(
        self, project_id: str, task_id: str, task_type: TaskType = TaskType.INFERENCE
    ) -> None:
        """Set task ID for a project based on task type."""
        project = self._get_project_or_404(project_id)
        key = (
            "task_id"
            if task_type == TaskType.INFERENCE
            else f"{task_type.value}_task_id"
        )
        project.parameters[key] = task_id
        flag_modified(project, "parameters")
        self.db.commit()

    def record_task_completion(
        self, project_id: str, task_type: TaskType, result_data: dict
    ) -> None:
        """Record task completion with all database updates in single transaction."""
        project = self._get_project_or_404(project_id)
        self.db.refresh(project)

        # Determine result metadata based on task type
        if task_type == TaskType.INFERENCE:
            result_key = "inference"
            file_key = "inference_key"
            file_check_key = "inference_file"
            model_id = result_data.get("model", "unknown")
            result_type = "image"
        elif task_type == TaskType.POLYGONIZE:
            result_key = "polygons"
            file_key = "polygon_key"
            file_check_key = "polygon_file"
            model_id = "polygonization"
            result_type = "geojson"
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        if result_data.get(file_check_key):
            # Create InferenceResult record
            file_path = result_data.get(file_key, result_data[file_check_key])
            inference_result = InferenceResult(
                project_id=project.id,
                model_id=model_id,
                result_type=result_type,
                file_path=file_path,
            )
            self.db.add(inference_result)

            if not project.results:
                project.results = {}

            project.results[result_key] = {
                "file_path": result_data[file_key],
                "metrics": {
                    k: v
                    for k, v in result_data.items()
                    if k not in [file_check_key, file_key]
                },
            }
            flag_modified(project, "results")

        project.status = ProjectStatus.COMPLETED
        self.db.commit()
        self.db.refresh(project)

    # --- Public API: Results & Configuration ---

    def get_inference_results(self, project_id: str) -> dict[str, Any]:
        """Get inference results for a completed project."""
        project = self._get_project_or_404(project_id)

        if project.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Project inference is not completed. "
                    f"Current status: {project.status}"
                ),
            )

        results = (
            self.db.query(InferenceResult)
            .filter(InferenceResult.project_id == project_id)
            .order_by(InferenceResult.created_at.desc())
            .all()
        )

        image_result = None
        geojson_result = None

        for result in results:
            if result.result_type == "image" and image_result is None:
                image_result = result
            elif result.result_type == "geojson" and geojson_result is None:
                geojson_result = result

            if image_result and geojson_result:
                break

        if not image_result and not geojson_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No inference results found for this project",
            )

        return {
            "image_result": image_result,
            "geojson_result": geojson_result,
        }

    async def get_inference_result_geojson(self, project_id: str) -> dict[str, Any]:
        """Download and return GeoJSON results for a project."""
        results = self.get_inference_results(project_id)
        geojson_result = results.get("geojson_result")

        if not geojson_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No GeoJSON results found for this project",
            )

        async with temp_files_context(f"geojson_{project_id}.json") as temp_files:
            temp_file = temp_files[0]
            await self.storage.download(geojson_result.file_path, temp_file)

            async with aiofiles.open(temp_file) as f:
                content = await f.read()
                return dict(json.loads(content))

    def get_inference_result_file_path(self, project_id: str) -> str:
        """Get file path for inference result image."""
        results = self.get_inference_results(project_id)
        image_result = results.get("image_result")

        if not image_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No image results found for this project",
            )

        return str(image_result.file_path)

    async def get_inference_results_response(
        self, project_id: str, content_type: str | None = None
    ) -> dict[str, Any]:
        """Get inference results formatted for API response."""
        results = self.get_inference_results(project_id)
        image_result = results.get("image_result")
        geojson_result = results.get("geojson_result")

        if content_type:
            if "geo+json" in content_type and geojson_result:
                geojson_data = await self.get_inference_result_geojson(project_id)
                return {
                    "data": geojson_data,
                    "media_type": "application/geo+json",
                    "response_type": "geojson",
                }
            elif "tiff" in content_type and image_result:
                file_path = self.get_inference_result_file_path(project_id)
                return {
                    "file_path": file_path,
                    "media_type": "image/tiff",
                    "filename": f"inference_{project_id}.tif",
                    "response_type": "file",
                }

        inference_url = await self._safe_get_url(
            image_result.file_path if image_result else None
        )
        polygons_url = await self._safe_get_url(
            geojson_result.file_path if geojson_result else None
        )

        response_data = {
            "inference": inference_url,
            "polygons": polygons_url,
        }

        return {
            "data": response_data,
            "media_type": "application/json",
            "response_type": "default",
        }

    def get_api_configuration(self) -> dict[str, Any]:
        """Get API configuration for root endpoint."""
        settings = get_settings()
        return {
            "api_version": settings.api.version,
            "title": settings.api.title,
            "description": settings.api.description,
            "min_area_km2": settings.processing.min_area_km2,
            "max_area_km2": settings.processing.max_area_km2,
            "models": settings.models,
        }

    # --- Internal Helper Methods ---

    def _get_project_or_404(self, project_id: str) -> Project:
        """Get project by ID or raise 404 HTTPException if not found."""
        project: Project | None = (
            self.db.query(Project).filter(Project.id == project_id).first()
        )
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found",
            )
        return project

    def _update_project_params(
        self, project_id: str, param_key: str, params: dict[str, Any]
    ) -> None:
        """Update project parameters and reset status to queued."""
        project = self._get_project_or_404(project_id)
        project.parameters[param_key] = params
        flag_modified(project, "parameters")
        project.status = ProjectStatus.QUEUED
        project.progress = None
        self.db.commit()

    async def _safe_get_url(self, file_path: str | None) -> str | None:
        """Safely get URL for file path, returning None on any error."""
        if not file_path:
            return None

        try:
            return await self.storage.get_url(file_path)
        except Exception as e:
            logger.warning(f"Could not generate URL for {file_path}: {e}")
            return None

    async def _get_project_results_urls(self, project: Project) -> ProjectResultLinks:
        """Convert database results to ProjectResults with proper URLs."""
        inference_url = None
        polygons_url = None

        if project.results:
            # Handle inference results
            if project.results.get("inference"):
                inference_data = project.results["inference"]
                if isinstance(inference_data, dict):
                    inference_url = await self._safe_get_url(
                        inference_data.get("file_path")
                    )

            # Handle polygon results
            if project.results.get("polygons"):
                polygons_data = project.results["polygons"]
                if isinstance(polygons_data, dict):
                    polygons_url = await self._safe_get_url(
                        polygons_data.get("file_path")
                    )

        return ProjectResultLinks(
            inference=inference_url,
            polygons=polygons_url,
        )

    async def _cleanup_project_files(self, project_id: str) -> None:
        """Delete all storage files for a project (TIF files, GeoJSON results, etc.)."""
        if not self.storage:
            logger.warning(
                "No storage backend configured, skipping file cleanup for project %s",
                project_id,
            )
            return

        try:
            project_prefix = f"projects/{project_id}/"
            files_to_delete = await self.storage.list_files(project_prefix)

            if not files_to_delete:
                logger.info(f"No files found to delete for project {project_id}")
                return

            deleted_count = 0
            for file_key in files_to_delete:
                try:
                    await self.storage.delete(file_key)
                    logger.info(f"Deleted storage file: {file_key}")
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file_key}: {e}")
                    # Continue deleting other files

            logger.info(
                "Cleanup completed for project %s: %d/%d files deleted",
                project_id,
                deleted_count,
                len(files_to_delete),
            )

        except Exception as e:
            logger.error(f"Failed to cleanup files for project {project_id}: {e}")
            # Don't fail the project deletion due to storage cleanup issues
