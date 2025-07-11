import contextlib
import time
from unittest.mock import patch

import pytest_asyncio
from app.core.task_manager import TaskManager


@pytest_asyncio.fixture
async def setup_task_manager():
    """Setup task manager for testing."""
    manager = TaskManager(max_workers=1)
    await manager.start_workers()
    yield manager
    with contextlib.suppress(RuntimeError, Exception):
        await manager.stop_workers()
    manager.active_tasks.clear()


def test_inference_endpoint_with_urls(client, setup_task_manager):
    """Test inference endpoint with URL-based processing."""
    response = client.post("/projects", json={"title": "URL Inference Test"})
    project_id = response.json()["id"]

    inference_params = {
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": ["https://example.com/image1.tif", "https://example.com/image2.tif"],
        "bbox": [13.0, 48.0, 13.05, 48.05],
        "resize_factor": 1.0,
    }

    response = client.put(f"/projects/{project_id}/inference", json=inference_params)

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert "project_id" in data
    assert data["status"] == "queued"

    project_response = client.get(f"/projects/{project_id}")
    assert project_response.json()["status"] == "queued"

    client.delete(f"/projects/{project_id}")


def test_inference_endpoint_validation_errors(client):
    """Test inference endpoint with invalid parameters."""
    response = client.post("/projects", json={"title": "Invalid Params Test"})
    project_id = response.json()["id"]

    response = client.put(
        f"/projects/{project_id}/inference",
        json={
            "images": [
                "https://example.com/image1.tif",
                "https://example.com/image2.tif",
            ],
            "resize_factor": 1.0,
        },
    )
    assert response.status_code == 400
    assert "field required" in response.json()["detail"].lower()

    response = client.put(
        f"/projects/{project_id}/inference",
        json={
            "model": "non_existent_model",
            "images": [
                "https://example.com/image1.tif",
                "https://example.com/image2.tif",
            ],
            "resize_factor": 1.0,
        },
    )
    assert response.status_code == 400

    client.delete(f"/projects/{project_id}")


def test_polygonize_endpoint(client, setup_task_manager):
    """Test polygonize endpoint."""
    response = client.post("/projects", json={"title": "Polygonize Test"})
    project_id = response.json()["id"]

    polygonize_params = {"simplify": 10, "min_size": 200, "close_interiors": True}

    response = client.put(f"/projects/{project_id}/polygons", json=polygonize_params)

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert "project_id" in data
    assert data["status"] == "queued"

    client.delete(f"/projects/{project_id}")


def test_project_status_endpoint(client, setup_task_manager):
    """Test project status endpoint."""
    response = client.post("/projects", json={"title": "Status Test"})
    project_id = response.json()["id"]

    response = client.get(f"/projects/{project_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["project_id"] == project_id
    assert data["status"] == "created"
    assert data["progress"] is None

    inference_params = {
        "bbox": [0, 1, 2, 3],
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": ["https://example.com/image1.tif", "https://example.com/image2.tif"],
        "resize_factor": 1.0,
    }
    client.put(f"/projects/{project_id}/inference", json=inference_params)

    response = client.get(f"/projects/{project_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"

    client.delete(f"/projects/{project_id}")


def test_task_status_endpoint(client, setup_task_manager):
    """Test individual task status endpoint."""
    response = client.post("/projects", json={"title": "Task Status Test"})
    project_id = response.json()["id"]

    inference_params = {
        "bbox": [0, 1, 2, 3],
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": ["https://example.com/image1.tif", "https://example.com/image2.tif"],
        "resize_factor": 1.0,
    }
    task_response = client.put(
        f"/projects/{project_id}/inference", json=inference_params
    )
    task_id = task_response.json()["task_id"]

    response = client.get(f"/projects/{project_id}/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["project_id"] == project_id
    assert data["status"] in ["pending", "running", "completed", "failed"]
    assert "created_at" in data

    response = client.get(f"/projects/{project_id}/tasks/non-existent")
    assert response.status_code == 404

    client.delete(f"/projects/{project_id}")


def test_inference_endpoint_non_existent_project(client):
    """Test inference endpoint with non-existent project."""
    inference_params = {
        "bbox": [0, 1, 2, 3],
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": ["https://example.com/image1.tif"],
        "resize_factor": 1.0,
    }

    response = client.put("/projects/non-existent/inference", json=inference_params)
    assert response.status_code == 404


@patch("app.core.processing.run_project_inference")
def test_end_to_end_workflow_mocked(mock_inference, client, setup_task_manager):
    """Test complete workflow with mocked processing."""
    mock_inference.return_value = {
        "inference_file": "/path/to/inference.tif",
        "total_time_ms": 1000,
    }

    response = client.post("/projects", json={"title": "E2E Test"})
    project_id = response.json()["id"]

    inference_params = {
        "bbox": [0, 1, 2, 3],
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": ["https://example.com/image1.tif", "https://example.com/image2.tif"],
        "resize_factor": 1.0,
    }

    response = client.put(f"/projects/{project_id}/inference", json=inference_params)
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    max_attempts = 10
    task_data = None
    for _ in range(max_attempts):
        time.sleep(0.1)
        response = client.get(f"/projects/{project_id}/tasks/{task_id}")
        task_data = response.json()

        if task_data["status"] == "completed":
            break
        assert task_data["status"] in ["pending", "running"]

    assert (
        task_data and task_data["status"] == "completed"
    ), f"Task did not complete after {max_attempts} attempts"

    client.delete(f"/projects/{project_id}")


def test_concurrent_tasks(client, setup_task_manager):
    """Test submitting multiple tasks concurrently."""
    project_ids = []
    for i in range(3):
        response = client.post("/projects", json={"title": f"Concurrent Test {i}"})
        project_ids.append(response.json()["id"])

    task_ids = []
    for project_id in project_ids:
        inference_params = {
            "bbox": [0, 1, 2, 3],
            "model": "2_Class_FULL_FTW_Pretrained",
            "images": [
                "https://example.com/image1.tif",
                "https://example.com/image2.tif",
            ],
            "resize_factor": 1.0,
        }
        response = client.put(
            f"/projects/{project_id}/inference", json=inference_params
        )
        assert response.status_code == 202
        task_ids.append(response.json()["task_id"])

    assert len(task_ids) == 3
    assert len(set(task_ids)) == 3

    for project_id in project_ids:
        client.delete(f"/projects/{project_id}")
