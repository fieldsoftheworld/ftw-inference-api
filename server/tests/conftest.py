from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from app.api.v1.dependencies import get_queue_service, get_storage_service
from app.core.auth import verify_auth
from app.core.config import StorageConfig
from app.core.queue import QueueBackend
from app.core.storage import LocalStorage
from app.db.database import create_tables
from app.main import app
from fastapi.testclient import TestClient
from moto import mock_aws


def override_verify_auth():
    """Override the auth dependency for testing."""
    return {"sub": "test_user"}


@pytest.fixture(scope="function")
def dynamodb_tables():
    """Create mock DynamoDB tables for testing."""
    with mock_aws():
        create_tables()
        yield


@pytest_asyncio.fixture(scope="function")
async def mock_queue():
    """Create a mock queue backend for API tests."""
    mock_queue = MagicMock(spec=QueueBackend)
    mock_queue.submit = AsyncMock(return_value="test-task-id")
    mock_queue.get_status = AsyncMock()
    mock_queue.cancel = AsyncMock()

    return mock_queue


@pytest.fixture(scope="function")
def client(dynamodb_tables, mock_queue, tmp_path):
    """Create a test client with overridden dependencies."""
    app.dependency_overrides[verify_auth] = override_verify_auth
    app.dependency_overrides[get_storage_service] = lambda: LocalStorage(
        StorageConfig(backend="local", output_dir=str(tmp_path))
    )
    app.dependency_overrides[get_queue_service] = lambda: mock_queue

    with TestClient(app) as c:
        yield c


# Test data fixtures for model validation tests
@pytest.fixture
def sample_bbox():
    """Standard test bounding box (San Francisco Bay Area)."""
    return [-122.0, 37.0, -121.0, 38.0]


@pytest.fixture
def sample_image_urls():
    """Standard test image URLs."""
    return {
        "single": ["https://example.com/a.tif"],
        "dual": ["https://example.com/a.tif", "https://example.com/b.tif"],
    }


@pytest.fixture
def model_ids():
    """Standard test model IDs."""
    return {
        "single_window": "FTW_v2_3_Class_FULL_singleWindow",
        "dual_window": "FTW_v1_2_Class_FULL",
    }


@pytest.fixture
def create_test_project(client):
    """Factory fixture to create test projects."""

    def _create_project(title: str = "Test Project"):
        response = client.post("/v1/projects", json={"title": title})
        assert response.status_code == 201
        return response.json()["id"]

    return _create_project
