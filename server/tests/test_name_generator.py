import re

from app.utils.name_generator import ProjectNameGenerator, generate_project_id


class TestProjectNameGenerator:
    """Test the ProjectNameGenerator class."""

    def test_generate_returns_correct_format(self):
        """Test that generated names follow adjective-animal-suffix format."""
        name = ProjectNameGenerator.generate()

        # Should match pattern: word-word-4chars
        pattern = r"^[a-z]+-[a-z]+-[a-z0-9]{4}$"
        assert re.match(pattern, name), (
            f"Generated name '{name}' doesn't match expected format"
        )

    def test_generate_uses_predefined_words(self):
        """Test that generated names use words from the predefined lists."""
        name = ProjectNameGenerator.generate()
        adjective, animal, suffix = name.split("-")

        assert adjective in ProjectNameGenerator.ADJECTIVES
        assert animal in ProjectNameGenerator.ANIMALS
        assert len(suffix) == 4

    def test_generate_creates_unique_names(self):
        """Test that multiple generations produce different names (usually)."""
        names = {ProjectNameGenerator.generate() for _ in range(10)}

        assert len(names) >= 8, "Expected mostly unique names from 10 generations"

    def test_generate_project_id_function(self):
        """Test the convenience function wrapper."""
        project_id = generate_project_id()

        # Should follow same format as the class method
        pattern = r"^[a-z]+-[a-z]+-[a-z0-9]{4}$"
        assert re.match(pattern, project_id)
