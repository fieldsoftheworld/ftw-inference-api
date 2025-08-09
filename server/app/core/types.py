import datetime
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pendulum
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from app.core.queue import QueueBackend
    from app.core.storage import StorageBackend


class AppState(TypedDict):
    """Type-safe container for FastAPI application state."""

    queue: "QueueBackend"
    storage: "StorageBackend"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    INFERENCE = "inference"
    POLYGONIZE = "polygonize"


class ProjectStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PendulumDateTimeAnnotation:
    """Annotation class for pendulum.DateTime validation and serialization."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Create a schema that validates pendulum.DateTime objects."""

        def validate_from_str(value: str) -> pendulum.DateTime:
            """Parse ISO format datetime string to pendulum.DateTime object."""
            try:
                parsed = pendulum.parse(value)
                if not isinstance(parsed, pendulum.DateTime):
                    raise ValueError("Expected datetime string, got date or time")
                return parsed
            except Exception as e:
                raise ValueError(f"Invalid datetime format: {e}") from e

        def validate_from_datetime(value: datetime.datetime) -> pendulum.DateTime:
            """Convert standard datetime to pendulum.DateTime."""
            if isinstance(value, pendulum.DateTime):
                return value
            try:
                # If datetime has no timezone, assume UTC
                if value.tzinfo is None:
                    return pendulum.instance(value, tz="UTC")
                return pendulum.instance(value)
            except Exception as e:
                raise ValueError(f"Cannot convert to pendulum.DateTime: {e}") from e

        # Schema for various input types
        from_str_schema = core_schema.chain_schema([
            core_schema.str_schema(),
            core_schema.no_info_plain_validator_function(validate_from_str),
        ])

        from_datetime_schema = core_schema.chain_schema([
            core_schema.is_instance_schema(datetime.datetime),
            core_schema.no_info_plain_validator_function(validate_from_datetime),
        ])

        from_pendulum_schema = core_schema.is_instance_schema(pendulum.DateTime)

        return core_schema.union_schema([
            from_pendulum_schema,
            from_str_schema,
            from_datetime_schema,
        ])

    @classmethod
    def __get_json_schema__(
        cls, _source_type: Any, _handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Get JSON schema for pendulum.DateTime."""
        return {
            "type": "string",
            "format": "date-time",
            "description": "ISO8601 datetime with timezone (UTC)",
        }


class LocalStorageInfo(TypedDict):
    """Type for local storage backend information."""

    backend: Literal["local"]
    output_dir: str
    temp_dir: str


class S3StorageInfo(TypedDict):
    """Type for S3 storage backend information."""

    backend: Literal["s3"]
    bucket_name: str
    region: str | None


class SourceCoopStorageInfo(TypedDict):
    """Type for Source Coop storage backend information."""

    backend: Literal["source_coop"]
    endpoint_url: str
    bucket_name: str
    repository_path: str
    use_secrets_manager: bool
    use_direct_s3: bool


# Storage backend types
StorageBackendType = Literal["local", "s3", "source_coop"]

# Union type for all storage backend info
StorageBackendInfo = LocalStorageInfo | S3StorageInfo | SourceCoopStorageInfo

# Type for use in model fields
PendulumDateTime = Annotated[pendulum.DateTime, PendulumDateTimeAnnotation]
