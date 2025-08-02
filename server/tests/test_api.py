import re
from pathlib import Path

DATETIME_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"


def test_root_endpoint(client):
    """Test the root endpoint returns API metadata."""
    response = client.get("/v1/")
    assert response.status_code == 200
    data = response.json()
    assert "api_version" in data
    assert "title" in data
    assert "description" in data
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


def test_create_project(client):
    """Test creating a new project."""
    response = client.post("/v1/projects", json={"title": "Test Project"})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == "Test Project"
    assert data["status"] == "created"
    assert data["progress"] is None
    assert re.match(DATETIME_RE, data["created_at"])

    project_id = data["id"]
    delete_response = client.delete(f"/v1/projects/{project_id}")
    assert delete_response.status_code == 204


def test_get_projects(client):
    """Test getting a list of projects."""
    response1 = client.post("/v1/projects", json={"title": "Test Project 1"})
    project_id1 = response1.json()["id"]
    response2 = client.post("/v1/projects", json={"title": "Test Project 2"})
    project_id2 = response2.json()["id"]

    response = client.get("/v1/projects")
    assert response.status_code == 200
    data = response.json()
    assert "projects" in data
    assert len(data["projects"]) >= 2

    delete_response1 = client.delete(f"/v1/projects/{project_id1}")
    assert delete_response1.status_code == 204
    delete_response2 = client.delete(f"/v1/projects/{project_id2}")
    assert delete_response2.status_code == 204


def test_get_project(client):
    """Test getting a single project."""
    create_response = client.post("/v1/projects", json={"title": "Single Test Project"})
    project_id = create_response.json()["id"]

    response = client.get(f"/v1/projects/{project_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == project_id
    assert data["title"] == "Single Test Project"
    assert data["status"] == "created"
    assert data["progress"] is None
    assert re.match(DATETIME_RE, data["created_at"])

    delete_response = client.delete(f"/v1/projects/{project_id}")
    assert delete_response.status_code == 204


def test_get_nonexistent_project(client):
    """Test getting a non-existent project returns 404."""
    response = client.get("/v1/projects/nonexistent")
    assert response.status_code == 404


def test_delete_project(client, tmp_path):
    """Test deleting a project and verify files are cleaned up."""
    create_response = client.post("/v1/projects", json={"title": "Project to Delete"})
    assert create_response.status_code == 201
    project_id = create_response.json()["id"]

    project_dir = Path(tmp_path) / f"projects/{project_id}"
    project_dir.mkdir(parents=True)
    result_file = project_dir / "result.tif"
    metadata_file = project_dir / "metadata.json"
    result_file.touch()
    metadata_file.touch()
    assert result_file.exists()
    assert metadata_file.exists()

    get_response = client.get(f"/v1/projects/{project_id}")
    assert get_response.status_code == 200
    assert get_response.json()["title"] == "Project to Delete"

    delete_response = client.delete(f"/v1/projects/{project_id}")
    assert delete_response.status_code == 204

    get_response_after = client.get(f"/v1/projects/{project_id}")
    assert get_response_after.status_code == 404

    assert not result_file.exists()
    assert not metadata_file.exists()


def test_delete_nonexistent_project(client):
    """Test deleting a non-existent project returns 404."""
    response = client.delete("/v1/projects/nonexistent")
    assert response.status_code == 404


def test_upload_image_and_inference(client, tmp_path):
    """Test uploading images and running inference."""
    test_image_a = Path(tmp_path) / "test_image_a.tif"
    test_image_a.write_bytes(b"Mock TIF image data A")

    test_image_b = Path(tmp_path) / "test_image_b.tif"
    test_image_b.write_bytes(b"Mock TIF image data B")

    create_response = client.post("/v1/projects", json={"title": "Image Test Project"})
    project_id = create_response.json()["id"]

    with test_image_a.open("rb") as f:
        response = client.put(
            f"/v1/projects/{project_id}/images/a",
            files={"file": ("test_image_a.tif", f, "image/tiff")},
        )
    assert response.status_code == 201

    with test_image_b.open("rb") as f:
        response = client.put(
            f"/v1/projects/{project_id}/images/b",
            files={"file": ("test_image_b.tif", f, "image/tiff")},
        )
    assert response.status_code == 201
    inference_params = {
        "bbox": [0, 1, 2, 3],
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": None,
        "resize_factor": 2,
        "patch_size": 512,
        "padding": 32,
        "polygonization": {"simplify": 10, "min_size": 200, "close_interiors": True},
    }

    response = client.put(f"/v1/projects/{project_id}/inference", json=inference_params)
    assert response.status_code == 202


def test_inference_without_images(client):
    """Test running inference without uploading images."""
    create_response = client.post("/v1/projects", json={"title": "No Images Project"})
    project_id = create_response.json()["id"]

    inference_params = {
        "bbox": None,
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": ["https://example.com/image1.tif", "https://example.com/image2.tif"],
        "resize_factor": 2,
        "patch_size": 1024,
        "padding": 64,
        "polygonization": {"simplify": 15, "min_size": 500, "close_interiors": False},
    }

    response = client.put(f"/v1/projects/{project_id}/inference", json=inference_params)
    assert response.status_code == 202


def test_get_inference_results_not_completed(client):
    """Test getting inference results for a project that's not completed."""
    create_response = client.post(
        "/v1/projects", json={"title": "Not Completed Project"}
    )
    project_id = create_response.json()["id"]

    response = client.get(f"/v1/projects/{project_id}/inference")
    assert response.status_code == 400

    delete_response = client.delete(f"/v1/projects/{project_id}")
    assert delete_response.status_code == 204


def test_example_endpoint(client):
    """Test the example endpoint for small area computation."""
    request_data = {
        "inference": {
            "model": "2_Class_FULL_FTW_Pretrained",
            "images": [
                "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729",
                "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20210925T101019_R022_T33UUP_20210926T121923",
            ],
            "bbox": [13.0, 48.0, 13.05, 48.05],
        },
        "polygons": {},
    }

    response = client.put("/v1/example", json=request_data)
    assert response.status_code == 200
    assert response.headers.get("content-type") == "application/geo+json"

    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data


def test_example_endpoint_area_too_large(client):
    """Test the example endpoint with an area that's too large."""
    request_data = {
        "inference": {
            "model": "2_Class_FULL_FTW_Pretrained",
            "bbox": [0.0, 0.0, 1.0, 1.0],
        },
        "polygons": {},
    }

    response = client.put("/v1/example", json=request_data)
    assert response.status_code == 400
    assert "Area too large" in response.json()["detail"]


def test_example_endpoint_invalid_bbox(client):
    """Test the example endpoint with invalid bbox values (outside EPSG:4326 bounds)."""
    request_data = {
        "inference": {
            "model": "2_Class_FULL_FTW_Pretrained",
            "bbox": [-190.0, 0.0, -185.0, 1.0],
        },
        "polygons": {"simplify": 10, "min_size": 200, "close_interiors": False},
    }

    response = client.put("/v1/example", json=request_data)
    assert response.status_code == 400
    assert "Longitude values" in response.json()["detail"]

    request_data = {
        "inference": {
            "model": "2_Class_FULL_FTW_Pretrained",
            "bbox": [0.0, -95.0, 1.0, -92.0],
        },
        "polygons": {},
    }

    response = client.put("/v1/example", json=request_data)
    assert response.status_code == 400
    assert "Latitude values" in response.json()["detail"]

    request_data = {
        "inference": {
            "model": "2_Class_FULL_FTW_Pretrained",
            "bbox": [10.0, 10.0, 5.0, 5.0],
        },
        "polygons": {},
    }

    response = client.put("/v1/example", json=request_data)
    assert response.status_code == 400
    assert "min values must be less than max values" in response.json()["detail"]


def test_polygonize_endpoint(client, tmp_path):
    """Test polygonizing from existing inference results."""
    test_image_a = Path(tmp_path) / "test_image_a.tif"
    test_image_a.write_bytes(b"Mock TIF image data A")

    test_image_b = Path(tmp_path) / "test_image_b.tif"
    test_image_b.write_bytes(b"Mock TIF image data B")

    create_response = client.post(
        "/v1/projects", json={"title": "Polygonize Test Project"}
    )
    project_id = create_response.json()["id"]

    with test_image_a.open("rb") as f:
        response = client.put(
            f"/v1/projects/{project_id}/images/a",
            files={"file": ("test_image_a.tif", f, "image/tiff")},
        )
    assert response.status_code == 201

    with test_image_b.open("rb") as f:
        response = client.put(
            f"/v1/projects/{project_id}/images/b",
            files={"file": ("test_image_b.tif", f, "image/tiff")},
        )
    assert response.status_code == 201

    inference_params = {
        "bbox": [0, 1, 2, 3],
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": None,
        "resize_factor": 2,
        "patch_size": 512,
        "padding": 32,
    }

    client.put(f"/v1/projects/{project_id}/inference", json=inference_params)

    polygonize_params = {
        "bbox": [0, 1, 2, 3],
        "model": "2_Class_FULL_FTW_Pretrained",
        "images": None,
        "resize_factor": 2,
        "patch_size": 512,
        "padding": 32,
        "polygonization": {"simplify": 5, "min_size": 100, "close_interiors": True},
    }

    response = client.put(f"/v1/projects/{project_id}/polygons", json=polygonize_params)
    assert response.status_code == 202
