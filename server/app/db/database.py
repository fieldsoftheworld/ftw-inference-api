from app.db.models import Image, InferenceResult, Project
from app.db.protocols import DynamoTable

TABLES: tuple[type[DynamoTable], ...] = (Image, InferenceResult, Project)
LOCAL_CAPACITY = 1


def create_tables() -> None:
    """Create DynamoDB tables for local development."""
    for table in TABLES:
        if not table.exists():
            table.create_table(
                read_capacity_units=LOCAL_CAPACITY,
                write_capacity_units=LOCAL_CAPACITY,
                wait=True,
            )


def verify_tables() -> None:
    """Verify tables exist in production."""
    for table in TABLES:
        if not table.exists():
            raise RuntimeError(f"Table {table.Meta.table_name} does not exist")
