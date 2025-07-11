import json
import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.auth import verify_auth
from app.core.config import get_settings
from app.core.file_manager import get_project_file_manager
from app.core.limiter import delete_logs, setup_logs, timeout
from app.core.processing import (
    prepare_inference_params,
    prepare_polygon_params,
    run_example,
)
from app.core.task_manager import get_task_manager
from app.db.database import get_db
from app.models.project import Image, InferenceResult, Project
from app.schemas.project import (
    InferenceParameters,
    PolygonizationParameters,
    ProcessingParameters,
    ProjectCreate,
    ProjectResponse,
    ProjectsResponse,
    RootResponse,
)


def _get_project_or_404(db: Session, project_id: str) -> Project:
    """Get project by ID or raise 404 error."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found",
        )
    return project


def _build_task_info(task_info: dict[str, str], task_id: str) -> dict[str, str]:
    """Build task info response dictionary."""
    return {
        "task_id": task_id,
        "task_type": task_info["task_type"],
        "task_status": task_info["status"],
        "created_at": task_info["created_at"],
        "started_at": task_info["started_at"],
        "completed_at": task_info["completed_at"],
        "error": task_info["error"],
    }


router = APIRouter()


@router.get("/", response_model=RootResponse)
async def get_root():
    """
    Root endpoint that returns basic server capabilities
    """
    settings = get_settings()

    # Get configurable content from settings
    title = settings.api_title
    description = settings.api_description

    return {
        "api_version": settings.api_version,
        "title": title,
        "description": description,
        "min_area_km2": settings.min_area_km2,
        "max_area_km2": settings.max_area_km2,
        "models": settings.models,
    }


@router.put("/example")
@timeout()
async def example(
    params: ProcessingParameters,
    db: Session = Depends(get_db),
    auth: dict = Depends(verify_auth),
    accept: str | None = Header(None),
):
    """
    Compute polygons for a small area quickly
    """
    # Get settings to access max area configuration
    settings = get_settings()

    if params.inference is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inference parameters are required",
        )

    ndjson = accept and "application/x-ndjson" in accept

    # Validate parameters
    try:
        inference_params = prepare_inference_params(
            params.inference.model_dump(),
            require_bbox=True,
            require_image_urls=True,
            max_area=settings.max_area_km2,
        )
        polygon_params = prepare_polygon_params(
            params.polygons.model_dump() if params.polygons else {}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Process inference synchronously
    log = None
    try:
        log = setup_logs(db)
        if log is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is busy, try again later",
            )

        response = await run_example(
            inference_params, polygon_params, ndjson=ndjson, gpu=settings.gpu
        )
        if ndjson:
            return PlainTextResponse(
                content=response,
                media_type="application/x-ndjson",
            )
        else:
            return JSONResponse(
                content=response,
                media_type="application/geo+json",
            )
    except HTTPException as e:
        # Re-raise HTTP exceptions to preserve status code and detail
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e
    finally:
        delete_logs(db, log)


@router.post(
    "/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED
)
async def create_project(
    project_data: ProjectCreate,
    db: Session = Depends(get_db),
    auth: dict = Depends(verify_auth),
):
    """
    Create a new project
    """
    new_project = Project(title=project_data.title)

    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    return new_project


@router.get("/projects", response_model=ProjectsResponse)
async def get_projects(
    db: Session = Depends(get_db), auth: dict = Depends(verify_auth)
):
    """
    Get list of projects
    """
    projects = db.query(Project).all()

    clean_projects = []
    for project in projects:
        file_manager = get_project_file_manager(project.id)
        clean_results = await file_manager.get_project_results()

        clean_project = ProjectResponse(
            id=project.id,
            title=project.title,
            status=project.status,
            progress=project.progress,
            created_at=project.created_at,
            parameters=project.parameters,
            results=clean_results,
        )
        clean_projects.append(clean_project)

    return {"projects": clean_projects}


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str, db: Session = Depends(get_db), auth: dict = Depends(verify_auth)
):
    """
    Get project details
    """
    project = _get_project_or_404(db, project_id)

    file_manager = get_project_file_manager(project_id)
    clean_results = await file_manager.get_project_results()

    return ProjectResponse(
        id=project.id,
        title=project.title,
        status=project.status,
        progress=project.progress,
        created_at=project.created_at,
        parameters=project.parameters,
        results=clean_results,
    )


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str, db: Session = Depends(get_db), auth: dict = Depends(verify_auth)
):
    """
    Delete project
    """
    project = _get_project_or_404(db, project_id)

    file_manager = get_project_file_manager(project_id)

    import contextlib

    with contextlib.suppress(Exception):
        await file_manager.cleanup_all_files()

    db.delete(project)
    db.commit()


@router.put(
    "/projects/{project_id}/images/{window}", status_code=status.HTTP_201_CREATED
)
async def upload_image(
    project_id: str,
    window: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    auth: dict = Depends(verify_auth),
):
    """
    Upload a raster image to a project
    """
    if window not in ["a", "b"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Window must be 'a' or 'b'"
        )

    _get_project_or_404(db, project_id)
    file_manager = get_project_file_manager(project_id)

    # Create temporary file for upload
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        # Get S3 upload path and upload file
        s3_key = await file_manager.get_upload_path(window)
        s3_url = await file_manager.upload_file(temp_path, s3_key)
        file_path = s3_url
    finally:
        # Clean up temporary file
        temp_path.unlink(missing_ok=True)

    # Check if there's already an image for this window and update it
    existing_image = (
        db.query(Image)
        .filter(Image.project_id == project_id, Image.window == window)
        .first()
    )

    if existing_image:
        existing_image.file_path = str(file_path)
        db.commit()
        db.refresh(existing_image)
    else:
        # Create new image record
        new_image = Image(
            project_id=project_id, window=window, file_path=str(file_path)
        )
        db.add(new_image)
        db.commit()


@router.put("/projects/{project_id}/inference")
async def inference(
    project_id: str,
    params: InferenceParameters,
    db: Session = Depends(get_db),
    auth: dict = Depends(verify_auth),
):
    """
    Run inference on project images
    """
    project = _get_project_or_404(db, project_id)

    # Validate parameters
    try:
        inference_params = prepare_inference_params(params.model_dump())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Update project with parameters
    project.parameters["inference"] = inference_params
    flag_modified(project, "parameters")
    project.status = "queued"
    project.progress = None
    db.commit()

    # Submit inference task to background queue
    task_manager = get_task_manager()
    task_data = {
        "task_type": "inference",
        "project_id": project_id,
        "inference_params": inference_params,
    }
    task_id = await task_manager.submit_task(task_data)

    # Store task ID in project for tracking
    project.parameters["task_id"] = task_id
    flag_modified(project, "parameters")
    db.commit()

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "message": "Inference task submitted successfully",
            "task_id": task_id,
            "project_id": project_id,
            "status": "queued",
        },
    )


@router.put("/projects/{project_id}/polygons")
async def polygonize(
    project_id: str,
    params: PolygonizationParameters,
    db: Session = Depends(get_db),
    auth: dict = Depends(verify_auth),
):
    """
    Run polygponization on project images or existing inference results
    """
    project = _get_project_or_404(db, project_id)

    # Validate parameters
    try:
        polygon_params = prepare_polygon_params(params.model_dump())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Update project with parameters
    project.parameters["polygons"] = polygon_params
    flag_modified(project, "parameters")
    project.status = "queued"
    project.progress = None
    db.commit()

    # Submit polygonization task to background queue
    task_manager = get_task_manager()
    task_data = {
        "task_type": "polygonize",
        "project_id": project_id,
        "polygon_params": polygon_params,
    }
    task_id = await task_manager.submit_task(task_data)

    # Store task ID in project for tracking
    project.parameters["polygonize_task_id"] = task_id
    flag_modified(project, "parameters")
    db.commit()

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "message": "Polygonization task submitted successfully",
            "task_id": task_id,
            "project_id": project_id,
            "status": "queued",
        },
    )


@router.get("/projects/{project_id}/inference")
async def get_inference_results(
    project_id: str,
    db: Session = Depends(get_db),
    auth: dict = Depends(verify_auth),
    content_type: str | None = None,
):
    """
    Get inference results for a project
    """
    project = _get_project_or_404(db, project_id)

    # Check if project status is completed
    if project.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Project inference is not completed. Current status: {project.status}"
            ),
        )

    # Get the latest inference results
    image_result = None
    geojson_result = None

    # Query for both result types
    results = (
        db.query(InferenceResult)
        .filter(InferenceResult.project_id == project_id)
        .order_by(InferenceResult.created_at.desc())
        .all()
    )

    # Find the latest of each type
    for result in results:
        if result.result_type == "image" and image_result is None:
            image_result = result
        elif result.result_type == "geojson" and geojson_result is None:
            geojson_result = result

        # Break if we found both
        if image_result and geojson_result:
            break

    if not image_result and not geojson_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No inference results found for this project",
        )

    # Handle content negotiation based on Content-Type header
    if content_type:
        if "geo+json" in content_type and geojson_result:
            # Return GeoJSON format
            with open(geojson_result.file_path) as f:
                geojson_data = json.load(f)
            return JSONResponse(content=geojson_data, media_type="application/geo+json")
        elif "tiff" in content_type and image_result:
            # Return tiff image
            return FileResponse(
                path=image_result.file_path,
                media_type="image/tiff",
                filename=f"inference_{project_id}.tif",
            )

    # Default: Return JSON with URLs to the data
    response_data = {
        "inference": "https://host.example/inference.tif",
        "polygons": "https://host.example/polygons.json" if geojson_result else None,
    }

    return JSONResponse(content=response_data, media_type="application/json")


@router.get("/projects/{project_id}/status")
async def get_project_status(
    project_id: str, db: Session = Depends(get_db), auth: dict = Depends(verify_auth)
):
    """Get current project status and progress"""
    project = _get_project_or_404(db, project_id)

    response_data = {
        "project_id": project_id,
        "status": project.status,
        "progress": project.progress,
        "parameters": project.parameters,
    }

    task_manager = get_task_manager()
    if "task_id" in project.parameters:
        task_info = await task_manager.get_task_info(project.parameters["task_id"])
        if task_info:
            response_data["task"] = _build_task_info(
                task_info, project.parameters["task_id"]
            )

    if "polygonize_task_id" in project.parameters:
        poly_task_info = await task_manager.get_task_info(
            project.parameters["polygonize_task_id"]
        )
        if poly_task_info:
            response_data["polygonize_task"] = _build_task_info(
                poly_task_info, project.parameters["polygonize_task_id"]
            )

    return JSONResponse(content=response_data)


@router.get("/projects/{project_id}/tasks/{task_id}")
async def get_task_status(
    project_id: str,
    task_id: str,
    db: Session = Depends(get_db),
    auth: dict = Depends(verify_auth),
):
    """Get specific task status and details"""
    _get_project_or_404(db, project_id)

    # Get task information
    task_manager = get_task_manager()
    task_info = await task_manager.get_task_info(task_id)

    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found",
        )

    # Verify task belongs to this project
    if task_info["project_id"] != project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Task does not belong to the specified project",
        )

    return JSONResponse(
        content={
            "task_id": task_id,
            "project_id": project_id,
            "task_type": task_info["task_type"],
            "status": task_info["status"],
            "created_at": task_info["created_at"],
            "started_at": task_info["started_at"],
            "completed_at": task_info["completed_at"],
            "error": task_info["error"],
            "result": task_info["result"],
        }
    )


@router.get("/health")
async def health_check():
    """Health check endpoint for load balancer"""
    return {"status": "healthy"}
