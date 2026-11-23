import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import pendulum
from pynamodb.attributes import NumberAttribute, UnicodeAttribute, UTCDateTimeAttribute
from pynamodb.models import Model

from app.core.config import get_settings
from app.core.types import ProjectStatus
from app.utils.name_generator import generate_project_id

settings = get_settings()


class _TableMeta:
    """Base for PynamoDB ``Meta`` classes.

    Declares the members of ``pynamodb.models.MetaProtocol`` as class variables
    so type checkers accept the class object as a ``MetaProtocol``. Members
    without a value here stay unset, so PynamoDB still fills in its own
    defaults when the model class is created.
    """

    table_name: ClassVar[str]
    read_capacity_units: ClassVar[int | None]
    write_capacity_units: ClassVar[int | None]
    region: ClassVar[str | None] = settings.dynamodb.aws_region
    host: ClassVar[str | None] = settings.dynamodb.dynamodb_endpoint  # local dev
    connect_timeout_seconds: ClassVar[int]
    read_timeout_seconds: ClassVar[int]
    max_retry_attempts: ClassVar[int]
    max_pool_connections: ClassVar[int]
    extra_headers: ClassVar[Mapping[str, str]]
    aws_access_key_id: ClassVar[str | None]
    aws_secret_access_key: ClassVar[str | None]
    aws_session_token: ClassVar[str | None]
    billing_mode: ClassVar[str | None]
    tags: ClassVar[dict[str, str] | None]
    stream_view_type: ClassVar[str | None]


class _ProjectMeta(_TableMeta):
    table_name = f"{settings.dynamodb.table_prefix}projects"


class _ImageMeta(_TableMeta):
    table_name = f"{settings.dynamodb.table_prefix}images"


class _InferenceResultMeta(_TableMeta):
    table_name = f"{settings.dynamodb.table_prefix}inference-results"


class _FeedbackMeta(_TableMeta):
    table_name = f"{settings.dynamodb.table_prefix}feedback"


class Project(Model):
    Meta = _ProjectMeta

    id = UnicodeAttribute(hash_key=True, default_for_new=lambda: generate_project_id())
    title = UnicodeAttribute()
    status = UnicodeAttribute(default=ProjectStatus.CREATED.value)
    progress = UnicodeAttribute(null=True)  # Store as string
    created_at = UTCDateTimeAttribute(default_for_new=lambda: datetime.now(UTC))
    parameters = UnicodeAttribute(default="{}")
    results = UnicodeAttribute(default="{}")

    @property
    def parameters_dict(self) -> dict[str, Any]:
        parsed: dict[str, Any] = json.loads(self.parameters) if self.parameters else {}
        return parsed

    @parameters_dict.setter
    def parameters_dict(self, value: dict[str, Any]) -> None:
        self.parameters = json.dumps(value)

    @property
    def results_dict(self) -> dict[str, Any]:
        parsed: dict[str, Any] = json.loads(self.results) if self.results else {}
        return parsed

    @results_dict.setter
    def results_dict(self, value: dict[str, Any]) -> None:
        self.results = json.dumps(value)

    @property
    def created_at_pendulum(self) -> pendulum.DateTime:
        return pendulum.instance(self.created_at).in_timezone("UTC")


class Image(Model):
    Meta = _ImageMeta

    id = UnicodeAttribute(hash_key=True, default_for_new=lambda: str(uuid.uuid4()))
    project_id = UnicodeAttribute()
    window = UnicodeAttribute()  # 'a' or 'b'
    file_path = UnicodeAttribute()
    created_at = UTCDateTimeAttribute(default_for_new=lambda: datetime.now(UTC))

    @classmethod
    def get_by_project_and_window(cls, project_id: str, window: str) -> "Image | None":
        """Get image by project_id and window."""
        try:
            result = next(
                cls.scan((cls.project_id == project_id) & (cls.window == window))
            )
            return cast("Image", result)
        except StopIteration:
            return None


class InferenceResult(Model):
    Meta = _InferenceResultMeta

    id = UnicodeAttribute(hash_key=True, default_for_new=lambda: str(uuid.uuid4()))
    project_id = UnicodeAttribute()
    model_id = UnicodeAttribute()
    result_type = UnicodeAttribute()  # 'image' or 'geojson'
    file_path = UnicodeAttribute()
    created_at = UTCDateTimeAttribute(default_for_new=lambda: datetime.now(UTC))

    @classmethod
    def get_latest_by_project_and_type(
        cls, project_id: str, result_type: str
    ) -> "InferenceResult | None":
        """Get latest result by project_id and result_type."""
        results = list(
            cls.scan((cls.project_id == project_id) & (cls.result_type == result_type))
        )
        return max(results, key=lambda x: x.created_at) if results else None


class FeedbackRecord(Model):
    """Stores all user feedback and contribution submissions in a single table.

    feedback_type values:
      - 'tile_rating'  : quick 1-3 viewport rating
      - 'tell_us_more' : detailed quality feedback form
      - 'contribute'   : community contribution interest form

    bbox and resolution are only populated for tile_rating and tell_us_more records.
    payload stores the full serialised request body as a JSON string.
    """

    Meta = _FeedbackMeta

    id = UnicodeAttribute(hash_key=True, default_for_new=lambda: str(uuid.uuid4()))
    feedback_type = UnicodeAttribute()  # 'tile_rating' | 'tell_us_more' | 'contribute'
    created_at = UTCDateTimeAttribute(default_for_new=lambda: datetime.now(UTC))
    bbox = UnicodeAttribute(
        null=True
    )  # JSON "[minLng,minLat,maxLng,maxLat]" — omitted for contribute
    resolution = NumberAttribute(
        null=True
    )  # pixel resolution in metres — omitted for contribute
    source_ip = UnicodeAttribute(null=True)  # client IP for future dedup
    payload = UnicodeAttribute(default="{}")  # full JSON of the request body
