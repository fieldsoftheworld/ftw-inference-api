import pendulum
from sqlalchemy import Column, Integer

from app.db.database import Base
from app.db.utils.pendulum_types import PendulumDateTime


class ExampleRequestLog(Base):
    __tablename__ = "example_request_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(PendulumDateTime, default=lambda: pendulum.now("UTC"))
