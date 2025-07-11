from unittest.mock import patch

import pytest
import pytest_asyncio
from app.core.auth import verify_auth
from app.db.database import Base, get_db
from app.main import app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Override the get_db dependency for testing
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override the auth dependency for testing
async def override_verify_auth():
    return {"sub": "test_user"}


@pytest.fixture(scope="function")
def test_db():
    # Create the database
    Base.metadata.create_all(bind=engine)
    yield  # Run the tests
    Base.metadata.drop_all(bind=engine)


@pytest_asyncio.fixture(scope="function")
async def mock_file_manager():
    """Create a mock file manager for API tests that doesn't actually use S3"""
    from unittest.mock import AsyncMock, MagicMock

    from app.core.file_manager import S3FileManager

    # Create a mock file manager
    mock_manager = MagicMock(spec=S3FileManager)

    # Mock all async methods
    mock_manager.ensure_directories = AsyncMock()
    mock_manager.get_upload_path = AsyncMock(
        return_value="projects/test-project/uploads/a.tif"
    )
    mock_manager.get_result_path = AsyncMock(
        return_value="projects/test-project/results/test.inference.tif"
    )
    mock_manager.upload_file = AsyncMock(
        return_value="https://mock-s3-url.com/test-bucket/projects/test-project/uploads/a.tif"
    )
    mock_manager.download_file = AsyncMock()
    mock_manager.list_uploaded_images = AsyncMock(
        return_value={"a": "mock-url-a", "b": "mock-url-b"}
    )
    mock_manager.has_uploaded_images = AsyncMock(return_value=True)
    mock_manager.get_latest_inference_result = AsyncMock(
        return_value="projects/test-project/results/latest.inference.tif"
    )
    mock_manager.cleanup_temp_files = AsyncMock()
    mock_manager.cleanup_all_files = AsyncMock()
    mock_manager.get_presigned_url = AsyncMock(
        return_value="https://mock-presigned-url.com"
    )
    mock_manager.list_results = AsyncMock(return_value=["result1.tif", "result2.tif"])
    mock_manager.get_project_results = AsyncMock(
        return_value={
            "inference": "https://mock-inference-url.com",
            "polygons": "https://mock-polygons-url.com",
        }
    )

    # Mock sync methods
    mock_manager.get_temp_path = MagicMock(return_value="/tmp/test-file.txt")

    return mock_manager


@pytest.fixture(scope="function")
def client(test_db, mock_file_manager):
    # Override the dependencies
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_auth] = override_verify_auth

    # Mock the file manager for API tests
    def mock_get_project_file_manager(project_id: str):
        return mock_file_manager

    with (
        patch(
            "app.api.endpoints.get_project_file_manager",
            side_effect=mock_get_project_file_manager,
        ),
        TestClient(app) as c,
    ):
        yield c
