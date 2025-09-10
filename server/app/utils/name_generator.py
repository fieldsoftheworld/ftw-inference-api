"""Project name generator utility for creating human-readable project IDs."""

import random
import string
from typing import ClassVar


class ProjectNameGenerator:
    """Generates human-readable project names like 'rambling-tiger-d3ec'"""

    ADJECTIVES: ClassVar[list[str]] = [
        "rambling",
        "swift",
        "gentle",
        "mighty",
        "clever",
        "brave",
        "golden",
        "silver",
        "crimson",
        "azure",
        "emerald",
        "violet",
        "amber",
        "serene",
        "fierce",
        "noble",
        "mystic",
        "ancient",
        "modern",
        "bright",
        "dark",
        "shining",
        "glowing",
        "dancing",
        "soaring",
        "flowing",
        "wild",
        "calm",
        "bold",
        "quiet",
        "vivid",
        "subtle",
        "radiant",
        "graceful",
        "sturdy",
    ]

    ANIMALS: ClassVar[list[str]] = [
        "tiger",
        "eagle",
        "wolf",
        "bear",
        "lion",
        "fox",
        "hawk",
        "owl",
        "deer",
        "rabbit",
        "dolphin",
        "whale",
        "shark",
        "penguin",
        "falcon",
        "raven",
        "swan",
        "turtle",
        "horse",
        "elephant",
        "leopard",
        "cheetah",
        "jaguar",
        "panther",
        "lynx",
        "otter",
        "badger",
        "squirrel",
        "mongoose",
        "crane",
        "salmon",
        "cobra",
        "viper",
        "gecko",
    ]

    @classmethod
    def generate(cls) -> str:
        """Generate a readable project name in format: adjective-animal-suffix"""
        adjective = random.choice(cls.ADJECTIVES)
        animal = random.choice(cls.ANIMALS)
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{adjective}-{animal}-{suffix}"


def generate_unique_project_id() -> str:
    """Generate a unique project ID with database collision checking."""
    # Importing here to avoid circular import issues
    try:
        from pynamodb.exceptions import DoesNotExist

        from app.db.models import Project

        max_attempts = 20
        for _ in range(max_attempts):
            project_id = ProjectNameGenerator.generate()
            try:
                # Try to get the project - if it exists, this won't raise an exception
                Project.get(project_id)
                # If we get here, project exists, try again
                continue
            except DoesNotExist:
                # Project doesn't exist, this ID is unique
                return project_id

        # Fallback if all attempts failed
        return ProjectNameGenerator.generate()
    except Exception:
        # Fallback if any imports or database operations fail
        return ProjectNameGenerator.generate()
