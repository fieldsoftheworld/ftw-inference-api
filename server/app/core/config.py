import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator
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
    """API metadata configuration"""

    title: str = "Fields of the World - Inference API"
    description: str = "A service for field boundary inference from satellite images."
    version: str = "0.1.0"


class ServerConfig(BaseModel):
    """Server runtime configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    database_url: str = "sqlite:///./data/ftw_inference.db"


class CORSConfig(BaseModel):
    """CORS configuration"""

    origins: list[str] = ["*"]

    @field_validator("origins", mode="before")
    @classmethod
    def parse_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class SecurityConfig(BaseModel):
    """Security and authentication configuration"""

    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    auth_disabled: bool = False

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
    """ML processing configuration"""

    min_area_km2: float = 100.0
    max_area_km2: float = 500.0
    max_concurrent_examples: int = 10
    example_timeout: int = 60
    gpu: int | None = None


class LoggingConfig(BaseModel):
    """Logging configuration"""

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
    """AWS CloudWatch configuration"""

    enabled: bool = False
    log_group: str = "/ftw-inference-api"
    log_stream_prefix: str = "app"
    region: str = "us-west-2"
    send_interval: int = 30
    max_batch_size: int = 10


class S3Config(BaseModel):
    """S3 storage configuration"""

    enabled: bool = False
    bucket_name: str | None = None
    region: str = "us-west-2"
    presigned_url_expiry: int = 3600

    @field_validator("bucket_name")
    @classmethod
    def validate_bucket_name(cls, v: str | None, info: ValidationInfo) -> str | None:
        if info.data.get("enabled") and not v:
            raise ValueError("bucket_name is required when S3 is enabled")
        return v


class StorageConfig(BaseModel):
    """Local storage configuration"""

    output_dir: str = "data/results"
    temp_dir: str = "data/temp"
    max_file_size_mb: int = 100


class Settings(BaseSettings):
    """Application settings with nested configuration support"""

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
    s3: S3Config = Field(default_factory=S3Config)
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
    """Returns a cached instance of the Settings object"""
    try:
        settings = Settings()
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
