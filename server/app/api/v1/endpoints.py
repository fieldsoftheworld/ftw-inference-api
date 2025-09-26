from typing import Any

from fastapi import APIRouter, File, Header, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from app.schemas.requests import (
    CreateProjectRequest,
    ExampleWorkflowRequest,
    InferenceRequest,
    PolygonizationRequest,
    SceneSelectionRequest,
)
from app.schemas.responses import (
    HealthResponse,
    ProjectResponse,
    ProjectsResponse,
    ProjectStatusResponse,
    RootResponse,
    SceneSelectionResponse,
    TaskDetailsResponse,
    TaskSubmissionResponse,
)

from .dependencies import (
    AuthDep,
    InferenceServiceDep,
    ProjectServiceDep,
    TaskServiceDep,
)

router = APIRouter()


@router.get("/", response_model=RootResponse, status_code=status.HTTP_200_OK)
async def get_root(project_service: ProjectServiceDep) -> Any:
    """Get API configuration and available endpoints."""
    return project_service.get_api_configuration()


@router.put("/example", response_model=RootResponse, status_code=status.HTTP_200_OK)
async def example(
    params: ExampleWorkflowRequest,
    inference_service: InferenceServiceDep,
    auth: AuthDep,
    accept: str | None = Header(None),
) -> PlainTextResponse | JSONResponse:
    """Run example workflow with inference and polygonization."""
    response = await inference_service.run_example_workflow(
        {
            "inference": params.inference.model_dump() if params.inference else None,
            "polygons": params.polygons.model_dump() if params.polygons else {},
        },
        accept_header=accept,
    )

    if response["format"] == "ndjson":
        return PlainTextResponse(
            content=response["data"],
            media_type=response["media_type"],
        )
    else:
        return JSONResponse(
            content=response["data"],
            media_type=response["media_type"],
        )


@router.post(
    "/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED
)
async def create_project(
    project_data: CreateProjectRequest,
    project_service: ProjectServiceDep,
    auth: AuthDep,
) -> ProjectResponse:
    """Create a new project with the provided configuration."""
    return await project_service.create_project(project_data)


@router.get(
    "/projects", response_model=ProjectsResponse, status_code=status.HTTP_200_OK
)
async def get_projects(
    project_service: ProjectServiceDep,
    auth: AuthDep,
) -> ProjectsResponse:
    """Retrieve all projects for the authenticated user."""
    projects = await project_service.get_projects()
    return ProjectsResponse(projects=projects)


@router.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    status_code=status.HTTP_200_OK,
)
async def get_project(
    project_id: str,
    project_service: ProjectServiceDep,
    auth: AuthDep,
) -> ProjectResponse:
    """Retrieve a specific project by ID."""
    return await project_service.get_project(project_id)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    project_service: ProjectServiceDep,
    auth: AuthDep,
) -> None:
    """Delete a project and all associated data."""
    await project_service.delete_project(project_id)


@router.put(
    "/projects/{project_id}/images/{window}", status_code=status.HTTP_201_CREATED
)
async def upload_image(
    project_id: str,
    window: str,
    project_service: ProjectServiceDep,
    auth: AuthDep,
    file: UploadFile = File(...),
) -> None:
    """Upload satellite image for a specific project window."""
    await project_service.upload_image(project_id, window, file)


@router.put(
    "/projects/{project_id}/inference",
    response_model=TaskSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def inference(
    project_id: str,
    params: InferenceRequest,
    inference_service: InferenceServiceDep,
    auth: AuthDep,
) -> TaskSubmissionResponse:
    """Submit ML inference task for field boundary detection."""
    task_id = await inference_service.submit_project_inference_workflow(
        project_id, params.model_dump()
    )
    return TaskSubmissionResponse(
        message="Inference task submitted successfully",
        task_id=task_id,
        project_id=project_id,
        status="queued",
    )


@router.put(
    "/projects/{project_id}/polygons",
    response_model=TaskSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def polygonize(
    project_id: str,
    params: PolygonizationRequest,
    inference_service: InferenceServiceDep,
    auth: AuthDep,
) -> TaskSubmissionResponse:
    """Submit polygonization task to convert raster results to vector polygons."""
    task_id = await inference_service.submit_project_polygonize_workflow(
        project_id, params.model_dump()
    )
    return TaskSubmissionResponse(
        message="Polygonization task submitted successfully",
        task_id=task_id,
        project_id=project_id,
        status="queued",
    )


@router.get(
    "/projects/{project_id}/inference",
    response_model=None,
    status_code=status.HTTP_200_OK,
)
async def get_inference_results(
    project_id: str,
    project_service: ProjectServiceDep,
    auth: AuthDep,
    content_type: str | None = None,
) -> JSONResponse | FileResponse:
    """Retrieve inference results as GeoJSON or file download."""
    response = await project_service.get_inference_results_response(
        project_id, content_type
    )

    if response["response_type"] == "geojson":
        return JSONResponse(content=response["data"], media_type=response["media_type"])
    elif response["response_type"] == "file":
        return FileResponse(
            path=response["file_path"],
            media_type=response["media_type"],
            filename=response["filename"],
        )
    else:
        return JSONResponse(content=response["data"], media_type=response["media_type"])


@router.get(
    "/projects/{project_id}/status",
    response_model=ProjectStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_project_status(
    project_id: str,
    project_service: ProjectServiceDep,
    task_service: TaskServiceDep,
    auth: AuthDep,
) -> ProjectStatusResponse:
    """Get comprehensive project status including task progress."""
    return await project_service.get_complete_project_status(project_id, task_service)


@router.get(
    "/projects/{project_id}/tasks/{task_id}",
    response_model=TaskDetailsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_task_status(
    project_id: str,
    task_id: str,
    task_service: TaskServiceDep,
    auth: AuthDep,
) -> TaskDetailsResponse:
    """Get detailed status and metadata for a specific task."""
    task_details = await task_service.get_task_details(project_id, task_id)
    return TaskDetailsResponse(**task_details)


@router.post(
    "/scene-selection",
    response_model=SceneSelectionResponse,
    status_code=status.HTTP_200_OK,
)
async def scene_selection(
    params: SceneSelectionRequest,
    inference_service: InferenceServiceDep,
    auth: AuthDep,
) -> SceneSelectionResponse:
    """Find optimal Sentinel-2 scenes for specified area and time."""
    result = await inference_service.run_scene_selection(params.model_dump())
    return SceneSelectionResponse(**result)


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    """Check API health status."""
    return HealthResponse(status="healthy")
