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


# NEW FIXTURE: Provides a single, shared DB session for a test function.
@pytest.fixture(scope="function")
def db_session():
    """Yield a new database session for a single test."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest_asyncio.fixture(scope="function")
async def mock_storage():
    """Create a mock storage backend for API tests that doesn't actually use S3"""
    from unittest.mock import AsyncMock, MagicMock

    from app.core.storage import StorageBackend

    # Create a mock storage backend
    mock_storage = MagicMock(spec=StorageBackend)

    # Mock all async methods
    mock_storage.upload = AsyncMock(
        return_value="https://mock-s3-url.com/test-bucket/projects/test-project/results/file.tif"
    )
    mock_storage.download = AsyncMock()
    mock_storage.delete = AsyncMock()
    mock_storage.exists = AsyncMock(return_value=True)
    mock_storage.get_presigned_url = AsyncMock(
        return_value="https://mock-presigned-url.com"
    )
    mock_storage.get_url = AsyncMock(return_value="https://mock-presigned-url.com")
    mock_storage.list_files = AsyncMock(return_value=["file1.tif", "file2.json"])

    return mock_storage


@pytest_asyncio.fixture(scope="function")
async def mock_queue():
    """Create a mock queue backend for API tests that doesn't actually queue tasks"""
    from unittest.mock import AsyncMock, MagicMock

    from app.core.queue import QueueBackend

    # Create a mock queue backend
    mock_queue = MagicMock(spec=QueueBackend)

    # Mock all async methods
    mock_queue.submit = AsyncMock(return_value="test-task-id")
    mock_queue.get_status = AsyncMock()
    mock_queue.cancel = AsyncMock()

    return mock_queue


@pytest.fixture(scope="function")
def client(test_db, mock_storage, mock_queue):
    from app.api.v1.dependencies import get_queue_service, get_storage_service

    # Override the dependencies
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_auth] = override_verify_auth
    app.dependency_overrides[get_storage_service] = lambda: mock_storage
    app.dependency_overrides[get_queue_service] = lambda: mock_queue

    with TestClient(app) as c:
        yield c
