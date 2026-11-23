"""
Protocol required because PynamoDBs metaclass design doesn't work well with MyPy.
"""

from typing import Protocol


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
    def create_table(
        cls,
        wait: bool = False,
        read_capacity_units: int | None = None,
        write_capacity_units: int | None = None,
    ) -> object:
        """Create the table with specified parameters."""
        ...
