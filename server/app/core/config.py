import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

logger = logging.getLogger(__name__)

# Find the project root directory (where .env file is located)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class APIConfig(BaseModel):
    """API metadata configuration."""

    title: str = "Fields of the World - Inference API"
    description: str = "A service for field boundary inference from satellite images."
    version: str = "0.1.0"


class ServerConfig(BaseModel):
    """Server runtime configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class CORSConfig(BaseModel):
    """CORS configuration."""

    origins: list[str] = ["*"]

    @field_validator("origins", mode="before")
    @classmethod
    def parse_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class SecurityConfig(BaseModel):
    """Security and authentication configuration."""

    secret_key: str = "secret_key"  # Default for local dev
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    auth_disabled: bool = True  # Default to disabled for dev

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "secret_key is required. "
                "Please set SECURITY__SECRET_KEY environment variable."
            )
        # Allow "secret_key" for backward compatibility with existing tokens
        if v != "secret_key" and len(v) < 32:
            raise ValueError("secret_key must be at least 32 characters long")
        return v


class ProcessingConfig(BaseModel):
    """ML processing configuration."""

    # Area limits for different processing modes
    min_area_km2: float = 0.1
    example_max_area_km2: float = 500.0  # Deprecated example endpoint
    project_max_area_km2: float = 3000.0  # Project/batch processing
    small_area_threshold_km2: float = 300.0

    @property
    def max_area_km2(self) -> float:
        return self.example_max_area_km2

    # Example endpoint deprecation
    example_enabled: bool = False

    max_concurrent_examples: int = 10
    example_timeout: int = 60
    gpu: int | None = None

    def get_max_area_for_mode(self, is_project: bool) -> float:
        """Get maximum area limit based on processing mode."""
        return self.project_max_area_km2 if is_project else self.example_max_area_km2


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        valid_formats = ["json", "text"]
        if v not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
        return v


class CloudWatchConfig(BaseModel):
    """AWS CloudWatch configuration."""

    enabled: bool = False
    log_group: str = "/ftw-inference-api"
    log_stream_prefix: str = "app"
    region: str = "us-west-2"
    send_interval: int = 30
    max_batch_size: int = 10


class DynamoDBConfig(BaseModel):
    """DynamoDB configuration."""

    aws_region: str = "us-west-2"
    dynamodb_endpoint: str | None = None  # For local development
    table_prefix: str = "ftw-"


class SourceCoopConfig(BaseModel):
    """Source Coop configuration."""

    bucket_name: str = "ftw"
    region: str = "us-west-2"
    endpoint_url: str = "https://data.source.coop"
    repository_path: str = "ftw-inference-output"
    access_key_id: str | None = None
    secret_access_key: str | None = None
    use_secrets_manager: bool = False
    secret_name: str = "ftw/source-coop/api-credentials"
    secrets_manager_region: str | None = None

    # TEMPORARY: STS workaround (Remove when Source Coop is fixed)
    use_sts_workaround: bool = True  # flag to enable/disable STS workaround
    sts_role_arn: str = "arn:aws:iam::417712557820:role/ftw-cross-account-access"
    sts_external_id: str = "tge"
    sts_real_bucket: str = "us-west-2.opendata.source.coop"
    sts_bucket_prefix: str = "ftw"  # Prefix added to all keys in real bucket


class StorageConfig(BaseModel):
    """Unified storage configuration."""

    backend: Literal["local", "source_coop"] = "local"
    output_dir: str = "data/results"
    max_file_size_mb: int = 100
    source_coop: SourceCoopConfig = Field(default_factory=SourceCoopConfig)


class Settings(BaseSettings):
    """Application settings with nested configuration support."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        toml_file=["config/base.toml"],
        validate_default=True,
        extra="allow",
    )

    api: APIConfig = Field(default_factory=APIConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cloudwatch: CloudWatchConfig = Field(default_factory=CloudWatchConfig)
    dynamodb: DynamoDBConfig = Field(default_factory=DynamoDBConfig)

    storage: StorageConfig = Field(default_factory=StorageConfig)

    models: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    try:
        settings = Settings()
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
