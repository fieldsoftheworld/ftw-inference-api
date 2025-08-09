"""
Protocol required because PynamoDBs metaclass design doesn't work well with MyPy.
"""

from typing import Any, Protocol


class DynamoTableMeta(Protocol):
    """Protocol for DynamoDB table Meta class."""

    table_name: str


class DynamoTable(Protocol):
    """Protocol for DynamoDB table classes."""

    Meta: type[DynamoTableMeta]

    @classmethod
    def exists(cls) -> bool:
        """Check if the table exists."""
        ...

    @classmethod
    def create_table(cls, **kwargs: Any) -> Any:
        """Create the table with specified parameters."""
        ...
