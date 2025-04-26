import pytest
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


@pytest.fixture(scope="function")
def client(test_db):
    # Override the dependencies
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_auth] = override_verify_auth

    # Create test client
    with TestClient(app) as c:
        yield c
