import json
import uuid
from datetime import datetime
from typing import Any, cast

import pendulum
from pynamodb.attributes import UnicodeAttribute, UTCDateTimeAttribute
from pynamodb.models import Model

from app.core.config import get_settings
from app.core.types import ProjectStatus
from app.utils.name_generator import generate_unique_project_id

settings = get_settings()


class Project(Model):
    class Meta:
        table_name = f"{settings.dynamodb.table_prefix}projects"
        region = settings.dynamodb.aws_region
        host = settings.dynamodb.dynamodb_endpoint  # For local development

    id = UnicodeAttribute(
        hash_key=True, default_for_new=lambda: generate_unique_project_id()
    )
    title = UnicodeAttribute()
    status = UnicodeAttribute(default=ProjectStatus.CREATED.value)
    progress = UnicodeAttribute(null=True)  # Store as string
    created_at = UTCDateTimeAttribute(default_for_new=datetime.utcnow)
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
    class Meta:
        table_name = f"{settings.dynamodb.table_prefix}images"
        region = settings.dynamodb.aws_region
        host = settings.dynamodb.dynamodb_endpoint

    id = UnicodeAttribute(hash_key=True, default_for_new=lambda: str(uuid.uuid4()))
    project_id = UnicodeAttribute()
    window = UnicodeAttribute()  # 'a' or 'b'
    file_path = UnicodeAttribute()
    created_at = UTCDateTimeAttribute(default_for_new=datetime.utcnow)

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
    class Meta:
        table_name = f"{settings.dynamodb.table_prefix}inference-results"
        region = settings.dynamodb.aws_region
        host = settings.dynamodb.dynamodb_endpoint

    id = UnicodeAttribute(hash_key=True, default_for_new=lambda: str(uuid.uuid4()))
    project_id = UnicodeAttribute()
    model_id = UnicodeAttribute()
    result_type = UnicodeAttribute()  # 'image' or 'geojson'
    file_path = UnicodeAttribute()
    created_at = UTCDateTimeAttribute(default_for_new=datetime.utcnow)

    @classmethod
    def get_latest_by_project_and_type(
        cls, project_id: str, result_type: str
    ) -> "InferenceResult | None":
        """Get latest result by project_id and result_type."""
        results = list(
            cls.scan((cls.project_id == project_id) & (cls.result_type == result_type))
        )
        return max(results, key=lambda x: x.created_at) if results else None
