import pendulum
from sqlalchemy import DateTime as SQLAlchemyDateTime
from sqlalchemy import TypeDecorator


class PendulumDateTime(TypeDecorator):
    """
    SQLAlchemy type that converts between SQLAlchemy DateTime and pendulum.DateTime
    """

    impl = SQLAlchemyDateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert pendulum.DateTime to a format compatible with SQLAlchemy"""
        if value is not None and isinstance(value, pendulum.DateTime):
            return value.in_timezone("UTC").naive()
        return value

    def process_result_value(self, value, dialect):
        """Convert database datetime to pendulum.DateTime"""
        if value is not None:
            # Assume the value from the database is UTC
            return pendulum.instance(value).in_timezone("UTC")
        return value
