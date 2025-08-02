from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import SourceCoopStorageConfig
from app.core.config_validator import ConfigValidator
from app.core.secrets import SecretsManager
from app.core.storage import SourceCoopStorage


class TestSourceCoopConfig:
    """Test Source Coop configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SourceCoopStorageConfig(bucket_name="test-bucket")
        assert config.backend == "source_coop"
        assert config.endpoint_url == "https://data.source.coop"
        assert config.bucket_name == "test-bucket"

    def test_enabled_config(self):
        """Test enabled configuration."""
        config = SourceCoopStorageConfig(
            bucket_name="test-bucket",
            access_key_id="test_key",
            secret_access_key="test_secret",
        )
        assert config.backend == "source_coop"
        assert config.access_key_id == "test_key"


class TestSourceCoopStorage:
    """Test Source Coop storage backend."""

    def test_init(self):
        """Test storage initialization."""
        config = SourceCoopStorageConfig(
            bucket_name="test-bucket",
            access_key_id="test_key",
            secret_access_key="test_secret",
        )
        storage = SourceCoopStorage(config)
        assert storage.config.bucket_name == "test-bucket"
        assert storage.config.endpoint_url == "https://data.source.coop"

    def test_get_storage_key(self):
        """Test storage key generation with repository path."""
        config = SourceCoopStorageConfig(
            bucket_name="test-bucket", repository_path="test-repo"
        )
        storage = SourceCoopStorage(config)

        key = storage._get_storage_key("test.txt")
        assert key == "test-repo/test.txt"

    def test_get_storage_key_no_prefix(self):
        """Test storage key generation without repository path."""
        config = SourceCoopStorageConfig(bucket_name="test-bucket", repository_path="")
        storage = SourceCoopStorage(config)

        key = storage._get_storage_key("test.txt")
        assert key == "test.txt"

    def test_strip_repository_path(self):
        """Test stripping repository path from storage key."""
        config = SourceCoopStorageConfig(
            bucket_name="test-bucket", repository_path="test-repo"
        )
        storage = SourceCoopStorage(config)

        key = storage._strip_repository_path("test-repo/subdir/file.txt")
        assert key == "subdir/file.txt"

    def test_strip_repository_path_no_prefix(self):
        """Test stripping when no repository path configured."""
        config = SourceCoopStorageConfig(bucket_name="test-bucket", repository_path="")
        storage = SourceCoopStorage(config)

        key = storage._strip_repository_path("subdir/file.txt")
        assert key == "subdir/file.txt"

    def test_lazy_initialization_not_called_immediately(self):
        """Test that credentials are not loaded during initialization."""
        config = SourceCoopStorageConfig(
            bucket_name="test-bucket",
            access_key_id="test_key",
            secret_access_key="test_secret",
        )
        storage = SourceCoopStorage(config)

        assert not storage._initialized


class TestSecretsManager:
    """Test AWS Secrets Manager integration."""

    @patch("app.core.secrets.aioboto3.Session")
    async def test_get_credentials_success(self, mock_session):
        """Test successful credential retrieval."""
        mock_client = AsyncMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": '{"access_key_id": "test_key", '
            '"secret_access_key": "test_secret"}'
        }
        mock_session.return_value.client.return_value.__aenter__.return_value = (
            mock_client
        )

        secrets_manager = SecretsManager()
        credentials = await secrets_manager.get_source_coop_credentials("test/secret")

        assert credentials["access_key_id"] == "test_key"
        assert credentials["secret_access_key"] == "test_secret"


class TestConfigValidator:
    """Test configuration validation."""

    async def test_source_coop_credential_validation(self):
        """Test Source Coop credential validation."""
        mock_settings = MagicMock()
        mock_storage = MagicMock()
        mock_storage.backend = "source_coop"
        mock_storage.use_secrets_manager = False
        mock_storage.access_key_id = None
        mock_storage.secret_access_key = None
        mock_settings.storage = mock_storage

        validator = ConfigValidator(mock_settings)
        validator._validate_source_coop_config()

        assert len(validator.errors) > 0
        assert any("access_key_id is required" in error for error in validator.errors)

    async def test_source_coop_validation_skipped_for_other_backends(self):
        """Test Source Coop validation is skipped for other backends."""
        mock_settings = MagicMock()
        mock_storage = MagicMock()
        mock_storage.backend = "local"
        mock_settings.storage = mock_storage

        validator = ConfigValidator(mock_settings)
        validator._validate_source_coop_config()

        assert len(validator.errors) == 0
