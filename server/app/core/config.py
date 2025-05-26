import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


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
