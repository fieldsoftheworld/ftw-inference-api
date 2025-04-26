import datetime
import enum
import uuid

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
)
from sqlalchemy.orm import relationship

from app.db.database import Base


class ProjectStatus(str, enum.Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, index=True)
    status = Column(String, default=ProjectStatus.CREATED)
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    parameters = Column(JSON, nullable=True)

    # Relationships
    images = relationship(
        "Image", back_populates="project", cascade="all, delete-orphan"
    )
    inference_results = relationship(
        "InferenceResult", back_populates="project", cascade="all, delete-orphan"
    )


class Image(Base):
    __tablename__ = "images"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    window = Column(String)  # 'a' or 'b'
    file_path = Column(String)  # Path to the stored image
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="images")


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"))
    model_id = Column(String)
    result_type = Column(String)  # 'image' or 'geojson'
    file_path = Column(String)  # Path to the stored result
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="inference_results")
