import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"  # json | text


class CloudWatchConfig(BaseModel):
    enabled: bool = False
    log_group: str = "/ftw-inference-api"
    log_stream_prefix: str = "app"
    region: str = "us-west-2"
    send_interval: int = 30
    max_batch_size: int = 10


class S3Config(BaseModel):
    enabled: bool = False
    bucket_name: str = "dev-ftw-api-model-outputs-2140860f"
    region: str = "us-west-2"
    presigned_url_expiry: int = 3600


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Fields of the World - Inference API"
    api_description: str = (
        "A service for field boundary inference from satellite images."
    )
    api_version: str = "0.1.0"

    # Server settings
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Minimum/Maximum area in square kilometers for the example endpoint
    min_area_km2: float = 100.0
    max_area_km2: float = 500.0
    # Concurrent request limit for the example endpoint
    max_concurrent_examples: int = 10
    # Time in seconds after which to time out and clean up pending example requests
    example_timeout: int = 60
    # GPU/CPU usage: None (CPU) or GPU index (e.g., 0 for the first GPU)
    gpu: int | None = None

    # Database settings
    database_url: str = "sqlite:///./data/ftw_inference.db"

    # CORS settings
    cors_origins: list[str] = ["*"]

    # Lift of available models
    models: list[dict[str, Any]] = Field(default_factory=list)

    # Security
    secret_key: str = "secret_key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    auth_disabled: bool = False  # Option to disable authentication

    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cloudwatch: CloudWatchConfig = Field(default_factory=CloudWatchConfig)

    # S3 Storage
    s3: S3Config = Field(default_factory=S3Config)

    def load_from_yaml(self, config_file: Path | str):
        """Load configuration from a YAML file"""
        config_path = Path(config_file)
        if not config_path.exists():
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Load API settings if they exist
        self.api_title = config.get("title", self.api_title)
        self.api_description = config.get("description", self.api_description)
        self.api_version = config.get("version", self.api_version)
        self.models = config.get("models", self.models)

        # Load server configuration values if they exist
        server_config = config.get("server", {})
        self.host = server_config.get("host", self.host)
        self.port = server_config.get("port", self.port)
        self.debug = server_config.get("debug", self.debug)
        self.cors_origins = server_config.get("cors", {}).get(
            "origins", self.cors_origins
        )
        self.database_url = server_config.get("database_url", self.database_url)

        security_config = config.get("security", {})
        self.secret_key = security_config.get("secret_key", self.secret_key)
        self.algorithm = security_config.get("algorithm", self.algorithm)
        self.access_token_expire_minutes = security_config.get(
            "access_token_expire_minutes", self.access_token_expire_minutes
        )
        self.auth_disabled = security_config.get("auth_disabled", self.auth_disabled)

        proc_config = config.get("processing", {})
        self.min_area_km2 = proc_config.get("min_area_km2", self.min_area_km2)
        self.max_area_km2 = proc_config.get("max_area_km2", self.max_area_km2)
        self.max_concurrent_examples = proc_config.get(
            "max_concurrent_examples", self.max_concurrent_examples
        )
        self.example_timeout = proc_config.get("example_timeout", self.example_timeout)
        self.gpu = proc_config.get("gpu", self.gpu)

        logging_config = config.get("logging", {})
        if logging_config:
            self.logging = LoggingConfig(**logging_config)

        cloudwatch_config = config.get("cloudwatch", {})
        if cloudwatch_config:
            self.cloudwatch = CloudWatchConfig(**cloudwatch_config)

        s3_config = config.get("s3", {})
        if s3_config:
            self.s3 = S3Config(**s3_config)


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    This ensures settings are loaded only once during the application lifecycle.
    """
    settings = Settings()

    config_path = os.environ.get("CONFIG_FILE", "config/config.yaml")
    p = Path(config_path)
    if p.exists():
        logger.info("Loading config from:" + str(p.absolute()))
        settings.load_from_yaml(p)
    else:
        logger.warning(f"Config file {p.absolute()} does not exist, using defaults.")

    return settings
