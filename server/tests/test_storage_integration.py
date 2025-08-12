"""Basic integration tests for storage system."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiofiles
import aiofiles.tempfile
import pytest
import pytest_asyncio
from app.core.config import (
    Settings,
    SourceCoopConfig,
    StorageConfig,
)
from app.core.storage import LocalStorage, SourceCoopStorage, get_storage
from app.services.project_service import ProjectService


class TestStorageFactory:
    """Test storage backend selection."""

    def test_get_storage_local_default(self):
        """Test that local storage is returned when explicitly configured."""
        # Create Settings without loading from environment or config files
        settings = Settings(_env_file=None, _env_ignore_empty=True)
        # Override storage to use local backend for this test
        settings.storage = StorageConfig(backend="local", output_dir="data/results")
        assert settings.storage.backend == "local"
        storage = get_storage(settings)
        assert isinstance(storage, LocalStorage)

    def test_get_storage_source_coop_enabled(self):
        """Test that Source Coop storage is returned when enabled."""
        settings = Settings(_env_file=None, _env_ignore_empty=True)
        settings.storage = StorageConfig(
            backend="source_coop",
            source_coop=SourceCoopConfig(
                bucket_name="test-bucket",
                access_key_id="test_key",
                secret_access_key="test_secret",
            ),
        )
        storage = get_storage(settings)
        assert isinstance(storage, SourceCoopStorage)


class TestLocalStorage:
    """Test local storage implementation."""

    @pytest_asyncio.fixture
    async def local_storage(self):
        """Create a local storage instance with temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = StorageConfig(output_dir=str(temp_dir))
            yield LocalStorage(storage_config)

    async def test_upload_download_cycle(self, local_storage):
        """Test basic upload/download functionality."""
        async with aiofiles.tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            await f.write("test content")
            test_file = Path(f.name)

        try:
            key = await local_storage.upload(test_file, "test/file.txt")
            assert key == "test/file.txt"

            assert await local_storage.file_exists("test/file.txt")

            download_path = test_file.parent / "downloaded.txt"
            await local_storage.download("test/file.txt", download_path)

            async with aiofiles.open(download_path) as f:
                content = await f.read()
            assert content == "test content"

            files = await local_storage.list_files("test/")
            assert "test/file.txt" in files

            url = await local_storage.get_url("test/file.txt")
            assert url.startswith("file://")

        finally:
            test_file.unlink(missing_ok=True)
            (test_file.parent / "downloaded.txt").unlink(missing_ok=True)


class TestProjectServiceWithStorage:
    """Test project service integration with storage backends."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        mock = MagicMock()
        mock.upload = AsyncMock(return_value="projects/test/file.tif")
        mock.download = AsyncMock()
        mock.get_url = AsyncMock(return_value="https://example.com/file.tif")
        mock.delete = AsyncMock()
        mock.list_files = AsyncMock(return_value=["projects/test/file.tif"])
        mock.file_exists = AsyncMock(return_value=True)
        return mock

    def test_project_service_storage_integration(self, mock_storage, mock_db):
        """Test that ProjectService correctly uses storage backend."""
        service = ProjectService(mock_storage)
        assert service.storage is mock_storage

    async def test_project_cleanup_calls_storage(self, mock_storage, mock_db):
        """Test that project cleanup calls storage methods."""
        service = ProjectService(mock_storage)

        mock_storage.list_files.return_value = [
            "projects/test-123/uploads/a/file1.tif",
            "projects/test-123/results/inference.tif",
        ]

        await service._cleanup_project_files("test-123")

        mock_storage.list_files.assert_called_once_with("projects/test-123/")
        assert mock_storage.delete.call_count == 2
