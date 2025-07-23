import uuid
from typing import Any

import pendulum
from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.types import ProjectStatus
from app.db.database import Base
from app.db.utils.pendulum_types import PendulumDateTime


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, index=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[ProjectStatus] = mapped_column(String, default=ProjectStatus.CREATED)
    progress: Mapped[float | None] = mapped_column(Float, default=None, nullable=True)
    created_at: Mapped[pendulum.DateTime] = mapped_column(
        PendulumDateTime, default=lambda: pendulum.now("UTC")
    )
    parameters: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default={"inference": None, "polygons": None}
    )
    results: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default={"inference": None, "polygons": None}
    )

    # Relationships
    images: Mapped[list["Image"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    inference_results: Mapped[list["InferenceResult"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Image(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, index=True, default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"))
    window: Mapped[str] = mapped_column(String)  # 'a' or 'b'
    file_path: Mapped[str] = mapped_column(String)  # Path to the stored image
    created_at: Mapped[pendulum.DateTime] = mapped_column(
        PendulumDateTime, default=lambda: pendulum.now("UTC")
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="images")


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, index=True, default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"))
    model_id: Mapped[str] = mapped_column(String)
    result_type: Mapped[str] = mapped_column(String)  # 'image' or 'geojson'
    file_path: Mapped[str] = mapped_column(String)  # Path to the stored result
    created_at: Mapped[pendulum.DateTime] = mapped_column(
        PendulumDateTime, default=lambda: pendulum.now("UTC")
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="inference_results")
