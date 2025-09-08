from unittest.mock import AsyncMock, patch

from app.core.config import SourceCoopConfig, StorageConfig
from app.core.secrets import SecretsManager
from app.core.storage import SourceCoopStorage


class TestSourceCoopConfig:
    """Test Source Coop configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        source_coop = SourceCoopConfig(bucket_name="test-bucket")
        config = StorageConfig(backend="source_coop", source_coop=source_coop)
        assert config.backend == "source_coop"
        assert config.source_coop.endpoint_url == "https://data.source.coop"
        assert config.source_coop.bucket_name == "test-bucket"

    def test_enabled_config(self):
        """Test enabled configuration."""
        source_coop = SourceCoopConfig(
            bucket_name="test-bucket",
            access_key_id="test_key",
            secret_access_key="test_secret",
        )
        config = StorageConfig(backend="source_coop", source_coop=source_coop)
        assert config.backend == "source_coop"
        assert config.source_coop.access_key_id == "test_key"


class TestSourceCoopStorage:
    """Test Source Coop storage backend."""

    def test_init(self):
        """Test storage initialization."""
        source_coop = SourceCoopConfig(
            bucket_name="test-bucket",
            access_key_id="test_key",
            secret_access_key="test_secret",
        )
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)
        assert storage.config.bucket_name == "test-bucket"
        assert storage.config.endpoint_url == "https://data.source.coop"

    def test_get_storage_key(self):
        """Test storage key generation with repository path."""
        source_coop = SourceCoopConfig(
            bucket_name="test-bucket", repository_path="test-repo"
        )
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)

        key = storage._get_storage_key("test.txt")
        assert key == "test-repo/test.txt"

    def test_get_storage_key_no_prefix(self):
        """Test storage key generation without repository path."""
        source_coop = SourceCoopConfig(bucket_name="test-bucket", repository_path="")
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)

        key = storage._get_storage_key("test.txt")
        assert key == "test.txt"

    def test_strip_repository_path(self):
        """Test stripping repository path from storage key."""
        source_coop = SourceCoopConfig(
            bucket_name="test-bucket", repository_path="test-repo"
        )
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)

        key = storage._strip_repository_path("test-repo/subdir/file.txt")
        assert key == "subdir/file.txt"

    def test_strip_repository_path_no_prefix(self):
        """Test stripping when no repository path configured."""
        source_coop = SourceCoopConfig(bucket_name="test-bucket", repository_path="")
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)

        key = storage._strip_repository_path("subdir/file.txt")
        assert key == "subdir/file.txt"

    def test_lazy_initialization_not_called_immediately(self):
        """Test that credentials are not loaded during initialization."""
        source_coop = SourceCoopConfig(
            bucket_name="test-bucket",
            access_key_id="test_key",
            secret_access_key="test_secret",
        )
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)

        assert not storage._initialized

    def test_init_with_iam_role(self):
        """Test storage initialization with STS workaround enabled."""
        source_coop = SourceCoopConfig(
            bucket_name="test-bucket",
            use_sts_workaround=True,
        )
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)
        assert storage.config.use_sts_workaround is True
        assert storage.config.bucket_name == "test-bucket"

    async def test_lazy_init_with_iam_role(self):
        """Test that STS workaround skips credential loading."""
        source_coop = SourceCoopConfig(
            bucket_name="test-bucket",
            use_sts_workaround=True,
        )
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)

        await storage._lazy_init()

        assert storage._initialized is True
        assert storage._access_key_id is None
        assert storage._secret_access_key is None

    @patch("app.core.storage.aioboto3.Session")
    async def test_s3_client_without_credentials(self, mock_session):
        """Test S3 client creation with IAM role (temporary cross-account fix)."""
        # Mock STS client for assume role
        mock_sts = AsyncMock()
        mock_sts.assume_role = AsyncMock(
            return_value={
                "Credentials": {
                    "AccessKeyId": "ASIA_TEMP_KEY",
                    "SecretAccessKey": "temp_secret",
                    "SessionToken": "temp_token",
                    "Expiration": "2025-01-01T00:00:00Z",
                }
            }
        )
        mock_sts_context = AsyncMock()
        mock_sts_context.__aenter__ = AsyncMock(return_value=mock_sts)
        mock_sts_context.__aexit__ = AsyncMock(return_value=None)

        # Mock S3 client
        mock_s3 = AsyncMock()
        mock_s3_context = AsyncMock()
        mock_s3_context.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_s3_context.__aexit__ = AsyncMock(return_value=None)

        # Configure session to return STS first, then S3
        mock_session.return_value.client.side_effect = [
            mock_sts_context,
            mock_s3_context,
        ]

        source_coop = SourceCoopConfig(
            bucket_name="test-bucket",
            use_sts_workaround=True,
        )
        storage_config = StorageConfig(backend="source_coop", source_coop=source_coop)
        storage = SourceCoopStorage(storage_config)

        async with storage._get_s3_client():
            # Verify assume role was called with correct parameters
            mock_sts.assume_role.assert_called_once_with(
                RoleArn="arn:aws:iam::417712557820:role/ftw-cross-account-access",
                RoleSessionName="ftw-source-coop-temp-access",
                ExternalId="tge",
            )

            # Verify S3 client was created with temporary credentials
            calls = mock_session.return_value.client.call_args_list
            s3_call = calls[1]
            assert s3_call.kwargs["aws_access_key_id"] == "ASIA_TEMP_KEY"
            assert s3_call.kwargs["aws_secret_access_key"] == "temp_secret"
            assert s3_call.kwargs["aws_session_token"] == "temp_token"


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
