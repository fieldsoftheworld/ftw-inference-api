import re

DATETIME_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"


def test_root_endpoint(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "api_version" in data
    assert "title" in data
    assert "description" in data
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


def test_create_project(client):
    """Test creating a new project"""
    response = client.post("/projects", json={"title": "Test Project"})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == "Test Project"
    assert data["status"] == "created"
    assert data["progress"] is None
    assert re.match(DATETIME_RE, data["created_at"])


def test_get_projects(client):
    """Test getting a list of projects"""
    # First create a project
    client.post("/projects", json={"title": "Test Project 1"})
    client.post("/projects", json={"title": "Test Project 2"})

    # Then get all projects
    response = client.get("/projects")
    assert response.status_code == 200
    data = response.json()
    assert "projects" in data
    assert len(data["projects"]) >= 2


def test_get_project(client):
    """Test getting a single project"""
    # First create a project
    create_response = client.post("/projects", json={"title": "Single Test Project"})
    project_id = create_response.json()["id"]

    # Then get the project
    response = client.get(f"/projects/{project_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == project_id
    assert data["title"] == "Single Test Project"
    assert data["status"] == "created"
    assert data["progress"] is None
    assert re.match(DATETIME_RE, data["created_at"])


def test_get_nonexistent_project(client):
    """Test getting a non-existent project"""
    response = client.get("/projects/nonexistent")
    assert response.status_code == 404


def test_upload_image_and_inference(client, tmp_path):
    """Test uploading images and running inference"""
    # Create a test image file
    test_image_a = tmp_path / "test_image_a.tif"
    test_image_a.write_bytes(b"Mock TIF image data A")

    test_image_b = tmp_path / "test_image_b.tif"
    test_image_b.write_bytes(b"Mock TIF image data B")

    # Create a project
    create_response = client.post("/projects", json={"title": "Image Test Project"})
    project_id = create_response.json()["id"]

    # Upload image for window A
    with open(test_image_a, "rb") as f:
        response = client.put(
            f"/projects/{project_id}/images/a",
            files={"file": ("test_image_a.tif", f, "image/tiff")},
        )
    assert response.status_code == 201

    # Upload image for window B
    with open(test_image_b, "rb") as f:
        response = client.put(
            f"/projects/{project_id}/images/b",
            files={"file": ("test_image_b.tif", f, "image/tiff")},
        )
    assert response.status_code == 201  # Run inference
    inference_params = {
        "bbox": [0, 1, 2, 3],  # Example bounding box
        "model": "default_model",
        "images": None,  # Use uploaded images
        "resize_factor": 1.5,
        "patch_size": 512,
        "padding": 32,
        "polygonization": {"simplify": 10, "min_size": 200, "close_interiors": True},
    }

    response = client.put(f"/projects/{project_id}/inference", json=inference_params)

    # Check that inference was queued for processing
    assert response.status_code == 202  # Accepted for processing
    data = response.json()
    assert "message" in data
    assert "queued" in data["message"].lower()

    if response.status_code == 200:
        # Either we get geojson content or a file response
        if response.headers.get("content-type") == "application/geo+json":
            data = response.json()
            assert data["type"] == "FeatureCollection"
            assert "features" in data
        else:
            # For file responses, we check the content-type and existence
            assert "image/" in response.headers.get("content-type", "")


def test_inference_without_images(client):
    """Test running inference without uploading images"""
    # Create a project
    create_response = client.post("/projects", json={"title": "No Images Project"})
    project_id = create_response.json()["id"]  # Run inference with image URLs
    inference_params = {
        "bbox": None,  # No bounding box
        "model": "default_model",
        "images": ["http://example.com/image1.tif", "http://example.com/image2.tif"],
        "resize_factor": 2,
        "patch_size": 1024,
        "padding": 64,
        "polygonization": {"simplify": 15, "min_size": 500, "close_interiors": False},
    }

    # Since we're mocking the HTTP requests in our implementation
    # this should actually queue for processing
    response = client.put(f"/projects/{project_id}/inference", json=inference_params)

    # We expect the request to be accepted and queued
    assert response.status_code == 202  # Accepted
    data = response.json()
    assert "message" in data
    assert "queued" in data["message"].lower()


def test_get_inference_results_not_completed(client):
    """Test getting inference results for a project that's not completed"""
    # Create a project
    create_response = client.post("/projects", json={"title": "Not Completed Project"})
    project_id = create_response.json()["id"]

    # Try to get inference results
    response = client.get(f"/projects/{project_id}/inference")
    assert response.status_code == 400  # Bad request, inference not completed


def test_example_endpoint(client):
    """Test the example endpoint for small area computation"""
    # Test data for the example endpoint
    request_data = {
        "inference": {
            "model": "default_model",
            "bbox": [10.0, 50.0, 10.01, 50.01],  # Small area
        },
        "polygons": {"simplify": 10, "min_size": 200, "close_interiors": False},
    }

    response = client.put("/example", json=request_data)
    assert response.status_code == 200
    assert response.headers.get("content-type") == "application/geo+json"

    # Check that the response is a valid GeoJSON
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) > 0


def test_example_endpoint_area_too_large(client):
    """Test the example endpoint with an area that's too large"""
    # Test data with a large area
    request_data = {
        "inference": {
            "model": "default_model",
            "bbox": [0.0, 0.0, 10.0, 10.0],  # Very large area
        },
        "polygons": {"simplify": 10, "min_size": 200, "close_interiors": False},
    }

    response = client.put("/example", json=request_data)
    assert response.status_code == 400  # Bad request, area too large


def test_polygonize_endpoint(client, tmp_path):
    """Test polygonizing from existing inference results"""
    # Create a test image file
    test_image_a = tmp_path / "test_image_a.tif"
    test_image_a.write_bytes(b"Mock TIF image data A")

    test_image_b = tmp_path / "test_image_b.tif"
    test_image_b.write_bytes(b"Mock TIF image data B")

    # Create a project
    create_response = client.post(
        "/projects", json={"title": "Polygonize Test Project"}
    )
    project_id = create_response.json()["id"]

    # Upload image for window A
    with open(test_image_a, "rb") as f:
        response = client.put(
            f"/projects/{project_id}/images/a",
            files={"file": ("test_image_a.tif", f, "image/tiff")},
        )
    assert response.status_code == 201

    # Upload image for window B
    with open(test_image_b, "rb") as f:
        response = client.put(
            f"/projects/{project_id}/images/b",
            files={"file": ("test_image_b.tif", f, "image/tiff")},
        )
    assert response.status_code == 201

    # Run inference first to create results that can be polygonized
    inference_params = {
        "bbox": [0, 1, 2, 3],
        "model": "default_model",
        "images": None,
        "resize_factor": 1.5,
        "patch_size": 512,
        "padding": 32,
    }

    response = client.put(f"/projects/{project_id}/inference", json=inference_params)

    # Now run polygonization with custom parameters
    polygonize_params = {
        "bbox": [0, 1, 2, 3],
        "model": "default_model",
        "images": None,
        "resize_factor": 1.5,
        "patch_size": 512,
        "padding": 32,
        "polygonization": {"simplify": 5, "min_size": 100, "close_interiors": True},
    }

    response = client.put(f"/projects/{project_id}/polygons", json=polygonize_params)
    assert response.status_code == 202  # Accepted for processing

    # Check the response indicates the task was queued
    data = response.json()
    assert "message" in data
    assert "queued" in data["message"].lower()
