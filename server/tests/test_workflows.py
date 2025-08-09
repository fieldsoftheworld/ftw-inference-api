from unittest.mock import AsyncMock, patch

import pendulum
import pytest
from app.api.v1.dependencies import get_storage_service
from app.core.queue import TaskInfo
from app.core.types import ProjectStatus, TaskStatus, TaskType
from app.services.project_service import ProjectService
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_full_inference_success_workflow(client: TestClient, dynamodb_tables):
    """Test complete inference workflow from task submission to result retrieval."""
    create_response = client.post(
        "/v1/projects", json={"title": "Workflow Success Test"}
    )
    assert create_response.status_code == 201
    project_id = create_response.json()["id"]

    inference_params = {
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": ["https://example.com/a.tif", "https://example.com/b.tif"],
    }
    submit_response = client.put(
        f"/v1/projects/{project_id}/inference", json=inference_params
    )
    assert submit_response.status_code == 202

    mock_result_data = {
        "inference_file": f"projects/{project_id}/results/inference_123.tif",
        "inference_key": f"projects/{project_id}/results/inference_123.tif",
        "model": "mock_model",
    }

    # Manually trigger completion since background tasks aren't running
    storage_mock = client.app.dependency_overrides[get_storage_service]()  # type: ignore[attr-defined]
    project_service = ProjectService(storage=storage_mock)
    project_service.record_task_completion(
        project_id, TaskType.INFERENCE, mock_result_data
    )

    status_response = client.get(f"/v1/projects/{project_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"

    with patch(
        "app.services.project_service.ProjectService._safe_get_url",
        new_callable=AsyncMock,
    ) as mock_get_url:
        mock_get_url.return_value = (
            "https://mock-storage.com/signed-url/inference_123.tif"
        )

        results_response = client.get(f"/v1/projects/{project_id}/inference")
        assert results_response.status_code == 200
        results_data = results_response.json()
        assert (
            results_data["inference"]
            == "https://mock-storage.com/signed-url/inference_123.tif"
        )


@pytest.mark.asyncio
async def test_task_failure_reporting(client: TestClient, mock_queue, dynamodb_tables):
    """Test that task failures are properly reported to the user."""
    create_response = client.post(
        "/v1/projects", json={"title": "Workflow Failure Test"}
    )
    project_id = create_response.json()["id"]
    submit_response = client.put(
        f"/v1/projects/{project_id}/inference",
        json={
            "model": "2_Class_FULL_FTW_Pretrained",
            "images": ["https://a.com/1.tif", "https://b.com/2.tif"],
        },
    )
    task_id = submit_response.json()["task_id"]

    error_message = "ML model file not found"
    failed_task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.FAILED,
        task_type="inference",
        created_at=pendulum.now(),
        updated_at=pendulum.now(),
        error=error_message,
    )
    mock_queue.get_status.return_value = failed_task_info

    storage_mock = client.app.dependency_overrides[get_storage_service]()  # type: ignore[attr-defined]
    project_service = ProjectService(storage=storage_mock)
    project_service.update_project_status(project_id, ProjectStatus.FAILED)

    status_response = client.get(f"/v1/projects/{project_id}/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "failed"
    assert status_data["task"]["task_status"] == "failed"
    assert status_data["task"]["error"] == error_message


@pytest.mark.asyncio
async def test_inference_with_invalid_model(client: TestClient):
    """Test that invalid model parameters are rejected at the API boundary."""
    create_response = client.post("/v1/projects", json={"title": "Invalid Model Test"})
    assert create_response.status_code == 201
    project_id = create_response.json()["id"]

    inference_params = {
        "model": "this-model-does-not-exist",
        "images": ["https://example.com/a.tif", "https://example.com/b.tif"],
    }
    submit_response = client.put(
        f"/v1/projects/{project_id}/inference", json=inference_params
    )

    assert submit_response.status_code == 400
    error_details = submit_response.json()
    assert "detail" in error_details
    error_str = str(error_details["detail"])
    assert "Input should be" in error_str and "Pretrained" in error_str
