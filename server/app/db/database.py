from collections.abc import Generator

from sqlalchemy import (
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import get_settings

# Create SQLAlchemy engine and base
engine = create_engine(
    get_settings().database_url, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_db_and_tables():
    """
    Create all tables in the database
    """
    Base.metadata.create_all(bind=engine)
