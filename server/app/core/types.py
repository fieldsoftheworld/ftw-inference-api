"""
Custom type adapters for Pydantic integration with Pendulum
"""

import datetime
from enum import Enum
from typing import Annotated, Any, Literal

import pendulum
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema


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


# Storage backend types
StorageBackendType = Literal["local", "s3", "source_coop"]

# Type for use in model fields
PendulumDateTime = Annotated[pendulum.DateTime, PendulumDateTimeAnnotation]
