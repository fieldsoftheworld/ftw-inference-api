from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


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

    # Root endpoint and models settings
    models_config: list[dict[str, Any]] = Field(default_factory=list)

    # Security
    secret_key: str = "change_this_in_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    auth_disabled: bool = False  # Option to disable authentication

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def load_from_yaml(self, config_file: Path | str):
        """Load configuration from a YAML file"""
        config_path = Path(config_file)
        if not config_path.exists():
            return

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Load API settings if they exist
        if "api" in config:
            api_config = config["api"]
            self.api_title = api_config.get("title", self.api_title)
            self.api_description = api_config.get("description", self.api_description)
            self.api_version = api_config.get("version", self.api_version)

        if "models" in config:
            self.models_config = config["models"]

        # Load server configuration values if they exist
        if "server" in config:
            server_config = config["server"]
            self.host = server_config.get("host", self.host)
            self.port = server_config.get("port", self.port)
            self.debug = server_config.get("debug", self.debug)

        if "cors" in config:
            self.cors_origins = config.get("cors", {}).get("origins", self.cors_origins)

        # Load security settings if they exist
        if "security" in config:
            security_config = config["security"]
            self.auth_disabled = security_config.get(
                "auth_disabled", self.auth_disabled
            )

        # Load database settings if they exist
        if "database" in config:
            database_config = config["database"]
            self.database_url = database_config.get("url", self.database_url)


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    This ensures settings are loaded only once during the application lifecycle.
    """
    settings = Settings()

    # Load configuration from default config file
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if config_path.exists():
        settings.load_from_yaml(config_path)

    return settings
