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


def generate_project_id() -> str:
    """Generate a project ID."""
    return ProjectNameGenerator.generate()
