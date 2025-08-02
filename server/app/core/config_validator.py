from pathlib import Path

from app.core.config import Settings, StorageConfig
from app.core.logging import get_logger
from app.core.secrets import get_secrets_manager
from app.core.types import StorageBackendInfo

logger = get_logger(__name__)


class ConfigurationValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigValidator:
    """Validates application configuration at startup."""

    def __init__(self, settings: Settings) -> None:
        """Initialize configuration validator."""
        self.settings = settings
        self.errors: list[str] = []

    async def validate_all(self) -> None:
        """Run all configuration validations."""
        logger.info("Starting configuration validation...")

        logger.info(f"Active storage backend: {self.settings.storage.backend}")

        self._validate_source_coop_config()
        await self._validate_aws_resources()
        self._validate_file_permissions()

        if self.errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in self.errors
            )
            logger.error(error_message)
            raise ConfigurationValidationError(error_message)

        logger.info("Configuration validation completed successfully")

    def _validate_source_coop_config(self) -> None:
        """Validate Source Coop specific configuration."""
        if self.settings.storage.backend != "source_coop":
            return

        config = self.settings.storage

        self._validate_direct_s3_config(config)
        self._validate_secrets_manager_config(config)
        self._warn_on_conflicting_credentials(config)

    def _validate_direct_s3_config(self, config: StorageConfig) -> None:
        """Validate direct S3 configuration settings."""
        if not config.use_direct_s3:
            return

        required_fields = [
            (
                "direct_s3_bucket",
                "direct_s3_bucket is required when use_direct_s3 is enabled",
            ),
            (
                "direct_s3_region",
                "direct_s3_region is required when use_direct_s3 is enabled",
            ),
        ]

        for field_name, error_msg in required_fields:
            if not getattr(config, field_name):
                self.errors.append(error_msg)

    def _validate_secrets_manager_config(self, config: StorageConfig) -> None:
        """Validate Secrets Manager or local credentials configuration."""
        if config.use_secrets_manager:
            required_fields = [
                (
                    "secret_name",
                    "secret_name is required when use_secrets_manager is enabled",
                ),
                (
                    "secrets_manager_region",
                    "secrets_manager_region is required "
                    "when use_secrets_manager is enabled",
                ),
            ]
        else:
            required_fields = [
                (
                    "access_key_id",
                    "access_key_id is required when use_secrets_manager is disabled",
                ),
                (
                    "secret_access_key",
                    "secret_access_key is required "
                    "when use_secrets_manager is disabled",
                ),
            ]

        for field_name, error_msg in required_fields:
            if not getattr(config, field_name):
                self.errors.append(error_msg)

    def _warn_on_conflicting_credentials(self, config: StorageConfig) -> None:
        """Warn if both Secrets Manager and local credentials are configured."""
        if config.use_secrets_manager and (
            config.access_key_id or config.secret_access_key
        ):
            logger.warning(
                "Both Secrets Manager and local credentials are configured. "
                "Secrets Manager will take precedence."
            )

    async def _validate_aws_resources(self) -> None:
        """Validate AWS resources and permissions."""
        if (
            self.settings.storage.backend != "source_coop"
            or not self.settings.storage.use_secrets_manager
        ):
            return

        config = self.settings.storage

        try:
            secrets_manager = get_secrets_manager(config.secrets_manager_region)
            await secrets_manager.get_source_coop_credentials(config.secret_name)
            logger.info(
                f"Successfully validated access to secret: {config.secret_name}"
            )

        except ValueError as e:
            if "not found" in str(e).lower():
                self.errors.append(
                    f"Secret '{config.secret_name}' not found in AWS Secrets Manager"
                )
            elif "access denied" in str(e).lower():
                self.errors.append(
                    f"Access denied to secret '{config.secret_name}'. "
                    "Check IAM permissions."
                )
            else:
                self.errors.append(
                    f"Failed to retrieve secret '{config.secret_name}': {e}"
                )

        except Exception as e:
            self.errors.append(f"Unexpected error validating AWS resources: {e}")

    def _validate_file_permissions(self) -> None:
        """Validate file permissions for local storage."""
        if hasattr(self.settings.storage, "output_dir"):
            storage_dir = Path(self.settings.storage.output_dir)
        else:
            storage_dir = Path("data/results")

        if hasattr(self.settings.storage, "temp_dir"):
            temp_dir = Path(self.settings.storage.temp_dir)
        else:
            temp_dir = Path("data/temp")

        for directory, name in [(storage_dir, "output_dir"), (temp_dir, "temp_dir")]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                test_file = directory / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                logger.debug(f"Validated write permissions for {name}: {directory}")

            except PermissionError:
                self.errors.append(f"No write permission for {name}: {directory}")
            except Exception as e:
                self.errors.append(f"Failed to validate {name} ({directory}): {e}")


async def validate_configuration(settings: Settings) -> None:
    """Validate application configuration at startup."""
    validator = ConfigValidator(settings)
    await validator.validate_all()


def get_storage_backend_info(settings: Settings) -> StorageBackendInfo:
    """Get information about the active storage backend."""
    storage_config = settings.storage

    if storage_config.backend == "source_coop":
        return {
            "backend": "source_coop",
            "endpoint_url": storage_config.endpoint_url,
            "bucket_name": storage_config.bucket_name,
            "repository_path": storage_config.repository_path,
            "use_secrets_manager": storage_config.use_secrets_manager,
            "use_direct_s3": storage_config.use_direct_s3,
        }
    elif storage_config.backend == "s3":
        return {
            "backend": "s3",
            "bucket_name": storage_config.bucket_name,
            "region": storage_config.region,
        }
    else:
        return {
            "backend": "local",
            "output_dir": storage_config.output_dir,
            "temp_dir": storage_config.temp_dir,
        }
