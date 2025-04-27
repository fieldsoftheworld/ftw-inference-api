import json
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.core.auth import verify_auth
from app.core.config import get_settings
from app.core.inference import process_inference_queue, run_inference_task
from app.db.database import get_db
from app.models.project import Image, InferenceResult, Project
from app.schemas.project import (
    InferenceParameters,
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
        "models": settings.models,
    }


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

    return {"message": "Image uploaded successfully"}


@router.put("/projects/{project_id}/inference")
async def run_inference(
    project_id: str,
    inference_params: InferenceParameters,
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

    # Update project with parameters
    project.parameters = inference_params.dict()
    db.commit()

    # If queue is True, queue the inference task and return status 202
    if inference_params.queue:
        project.status = "queued"
        project.progress = 0.0
        db.commit()

        # Start the inference task in the background
        run_inference_task(project_id, inference_params.dict())

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"message": "Inference task queued for processing"},
        )
    else:
        # Run inference synchronously and return results
        try:
            project.status = "running"
            db.commit()

            # Process inference
            result = process_inference_queue(project_id, inference_params.dict(), db)

            # Return appropriate response based on result type
            if result.result_type == "image":
                return FileResponse(
                    path=result.file_path,
                    media_type="image/tiff",
                    filename=f"inference_{project_id}.tif",
                )
            else:  # geojson
                with open(result.file_path) as f:
                    geojson_data = json.load(f)

                return JSONResponse(
                    content=geojson_data, media_type="application/geo+json"
                )

        except Exception as e:
            project.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            ) from e


@router.get("/projects/{project_id}/inference")
async def get_inference_results(
    project_id: str, db: Session = Depends(get_db), auth: dict = Depends(verify_auth)
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

    # Get the latest inference result
    latest_result = (
        db.query(InferenceResult)
        .filter(InferenceResult.project_id == project_id)
        .order_by(InferenceResult.created_at.desc())
        .first()
    )

    if not latest_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No inference results found for this project",
        )

    # Return appropriate response based on result type
    if latest_result.result_type == "image":
        return FileResponse(
            path=latest_result.file_path,
            media_type="image/tiff",
            filename=f"inference_{project_id}.tif",
        )
    else:  # geojson
        with open(latest_result.file_path) as f:
            geojson_data = json.load(f)

        return JSONResponse(content=geojson_data, media_type="application/geo+json")
