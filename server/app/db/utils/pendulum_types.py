from datetime import datetime
from typing import Any

import pendulum
from sqlalchemy import DateTime as SQLAlchemyDateTime
from sqlalchemy import TypeDecorator
from sqlalchemy.engine.interfaces import Dialect


class PendulumDateTime(TypeDecorator):
    """SQLAlchemy type that converts SQLAlchemy DateTime and pendulum.DateTime"""

    impl = SQLAlchemyDateTime
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> datetime | None:
        """Convert pendulum.DateTime to a format compatible with SQLAlchemy"""
        if value is not None and isinstance(value, pendulum.DateTime):
            naive_dt: datetime = value.in_timezone("UTC").naive()
            return naive_dt
        if isinstance(value, datetime):
            return value
        return None

    def process_result_value(
        self, value: Any, dialect: Dialect
    ) -> pendulum.DateTime | None:
        """Convert database datetime to pendulum.DateTime"""
        if value is not None:
            # Assume the value from the database is UTC
            return pendulum.instance(value).in_timezone("UTC")
        return None
