"""
Project name generator utility for creating human-readable project IDs.
"""

import random
import string
from collections.abc import Callable
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
        # Generate 4-character alphanumeric suffix (lowercase)
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{adjective}-{animal}-{suffix}"


def generate_unique_project_id(
    check_existence_func: Callable[[str], bool] | None = None,
) -> str:
    """
    Generate a unique project ID, checking against existing IDs.

    Args:
        check_existence_func: Function that takes an ID and returns True if it exists

    Returns:
        A unique project ID string

    Raises:
        RuntimeError: If unable to generate unique ID after max attempts
    """
    max_attempts = 20

    for _ in range(max_attempts):
        project_id = ProjectNameGenerator.generate()

        # If no check function provided, return the generated ID
        if check_existence_func is None:
            return project_id

        # Check if ID already exists
        if not check_existence_func(project_id):
            return project_id

    # If we get here, we've exhausted our attempts
    raise RuntimeError(
        f"Unable to generate unique project ID after {max_attempts} attempts"
    )
