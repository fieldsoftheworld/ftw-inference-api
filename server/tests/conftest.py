from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from app.api.v1.dependencies import get_queue_service, get_storage_service
from app.core.auth import verify_auth
from app.core.config import LocalStorageConfig
from app.core.queue import QueueBackend
from app.core.storage import LocalStorage
from app.db.database import Base, get_db
from app.main import app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override the get_db dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def override_verify_auth():
    """Override the auth dependency for testing."""
    return {"sub": "test_user"}


@pytest.fixture(scope="function")
def test_db():
    """Create and tear down test database."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Provide a single database session for a test function."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest_asyncio.fixture(scope="function")
async def mock_queue():
    """Create a mock queue backend for API tests."""
    mock_queue = MagicMock(spec=QueueBackend)
    mock_queue.submit = AsyncMock(return_value="test-task-id")
    mock_queue.get_status = AsyncMock()
    mock_queue.cancel = AsyncMock()

    return mock_queue


@pytest.fixture(scope="function")
def client(test_db, mock_queue, tmp_path):
    """Create a test client with overridden dependencies."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_auth] = override_verify_auth
    app.dependency_overrides[get_storage_service] = lambda: LocalStorage(
        LocalStorageConfig(output_dir=str(tmp_path))
    )
    app.dependency_overrides[get_queue_service] = lambda: mock_queue

    with TestClient(app) as c:
        yield c
