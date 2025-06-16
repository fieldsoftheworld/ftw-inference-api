import json
import os
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

from app.core.auth import verify_auth
from app.core.config import get_settings
from app.core.limiter import delete_logs, setup_logs, timeout
from app.core.processing import (
    prepare_inference_params,
    prepare_polygon_params,
    run_example,
)
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

# Create data directories if they don't exist
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")

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
    return {"projects": projects}


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str, db: Session = Depends(get_db), auth: dict = Depends(verify_auth)
):
    """
    Get project details
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found",
        )

    return project


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str, db: Session = Depends(get_db), auth: dict = Depends(verify_auth)
):
    """
    Delete project
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found",
        )

    db.delete(project)

    # Delete project directory if it exists
    project_dir = UPLOAD_DIR / project_id
    if project_dir.exists() and project_dir.is_dir():
        shutil.rmtree(project_dir)
    # Also delete results directory if it exists
    results_dir = RESULTS_DIR / project_id
    if results_dir.exists() and results_dir.is_dir():
        shutil.rmtree(results_dir)

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

    # Check if project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found",
        )

    # Create uploads directory structure if it doesn't exist
    project_dir = UPLOAD_DIR / project_id
    project_dir.mkdir(exist_ok=True, parents=True)

    # Save file
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".tif"
    file_path = project_dir / f"{window}{file_extension}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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
    # Check if project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found",
        )

    # Validate parameters
    try:
        inference_params = prepare_inference_params(params.model_dump())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Update project with parameters (no workaround needed)
    project.parameters["inference"] = inference_params
    project.status = "queued"
    project.progress = None
    db.commit()

    # Start the inference task in the background
    # TODO

    # return JSONResponse(status_code=status.HTTP_202_ACCEPTED)
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented"
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
    # Check if project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found",
        )

    # Validate parameters
    try:
        polygon_params = prepare_polygon_params(params.model_dump())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Update project with parameters (no workaround needed)
    project.parameters["polygons"] = polygon_params
    project.status = "queued"
    project.progress = None
    db.commit()

    # Start the polygonization task in the background
    # TODO

    # return JSONResponse(status_code=status.HTTP_202_ACCEPTED)
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented"
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
    # Check if project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID {project_id} not found",
        )

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
