import pytest
from app.core.file_manager import (
    ProjectFileManager,
    get_project_file_manager,
    validate_upload_file,
)


@pytest.fixture
def project_manager(tmp_path):
    """Create a project file manager with temporary directory."""
    # Store original directories for potential restoration
    _original_base_dirs = (
        ProjectFileManager.__module__ + ".UPLOAD_DIR",
        ProjectFileManager.__module__ + ".RESULTS_DIR",
        ProjectFileManager.__module__ + ".TEMP_DIR",
    )

    upload_dir = tmp_path / "uploads"
    results_dir = tmp_path / "results"
    temp_dir = tmp_path / "temp"

    manager = ProjectFileManager("test-project-123")
    manager.upload_dir = upload_dir / "test-project-123"
    manager.results_dir = results_dir / "test-project-123"
    manager.temp_dir = temp_dir / "test-project-123"

    yield manager


def test_project_file_manager_initialization():
    """Test ProjectFileManager initializes with correct paths."""
    manager = ProjectFileManager("test-project")

    assert manager.project_id == "test-project"
    assert "test-project" in str(manager.upload_dir)
    assert "test-project" in str(manager.results_dir)
    assert "test-project" in str(manager.temp_dir)


def test_ensure_directories(project_manager):
    """Test directory creation."""
    assert not project_manager.upload_dir.exists()
    assert not project_manager.results_dir.exists()
    assert not project_manager.temp_dir.exists()

    project_manager.ensure_directories()

    assert project_manager.upload_dir.exists()
    assert project_manager.results_dir.exists()
    assert project_manager.temp_dir.exists()


def test_get_upload_path(project_manager):
    """Test upload path generation."""
    path_a = project_manager.get_upload_path("a")
    path_b = project_manager.get_upload_path("b")

    assert path_a.name == "a.tif"
    assert path_b.name == "b.tif"
    assert str(project_manager.upload_dir) in str(path_a)

    with pytest.raises(ValueError, match="Window must be 'a' or 'b'"):
        project_manager.get_upload_path("c")


def test_get_result_path(project_manager):
    """Test result path generation."""
    inference_path = project_manager.get_result_path("inference")
    polygon_path = project_manager.get_result_path("polygons")

    assert inference_path.suffix == ".tif"
    assert "inference" in inference_path.name
    assert polygon_path.suffix == ".json"
    assert "polygons" in polygon_path.name

    with pytest.raises(ValueError, match="Unknown result type"):
        project_manager.get_result_path("unknown")


def test_get_temp_path(project_manager):
    """Test temporary path generation."""
    temp_path = project_manager.get_temp_path("test_file.txt")

    assert temp_path.name == "test_file.txt"
    assert str(project_manager.temp_dir) in str(temp_path)


def test_list_uploaded_images(project_manager):
    """Test listing uploaded images."""
    project_manager.ensure_directories()

    images = project_manager.list_uploaded_images()
    assert len(images) == 0

    (project_manager.upload_dir / "a.tif").write_bytes(b"test data a")
    (project_manager.upload_dir / "b.tif").write_bytes(b"test data b")

    images = project_manager.list_uploaded_images()
    assert len(images) == 2
    assert "a" in images
    assert "b" in images
    assert images["a"].exists()
    assert images["b"].exists()


def test_has_uploaded_images(project_manager):
    """Test checking for uploaded images."""
    project_manager.ensure_directories()

    assert not project_manager.has_uploaded_images()

    (project_manager.upload_dir / "a.tif").write_bytes(b"test data a")
    assert not project_manager.has_uploaded_images()

    (project_manager.upload_dir / "b.tif").write_bytes(b"test data b")
    assert project_manager.has_uploaded_images()


def test_get_latest_inference_result(project_manager):
    """Test getting latest inference result."""
    project_manager.ensure_directories()

    result = project_manager.get_latest_inference_result()
    assert result is None

    old_file = project_manager.results_dir / "old.inference.tif"
    new_file = project_manager.results_dir / "new.inference.tif"

    old_file.write_bytes(b"old data")
    new_file.write_bytes(b"new data")

    result = project_manager.get_latest_inference_result()
    assert result == new_file


def test_cleanup_temp_files(project_manager):
    """Test cleanup of temporary files."""
    project_manager.ensure_directories()

    temp_file = project_manager.temp_dir / "test.txt"
    temp_file.write_text("test content")
    assert temp_file.exists()

    project_manager.cleanup_temp_files()
    assert not project_manager.temp_dir.exists()


def test_cleanup_all_files(project_manager):
    """Test cleanup of all project files."""
    project_manager.ensure_directories()

    (project_manager.upload_dir / "test.txt").write_text("upload")
    (project_manager.results_dir / "test.txt").write_text("result")
    (project_manager.temp_dir / "test.txt").write_text("temp")

    project_manager.cleanup_all_files()

    assert not project_manager.upload_dir.exists()
    assert not project_manager.results_dir.exists()
    assert not project_manager.temp_dir.exists()


def test_validate_upload_file(tmp_path):
    """Test file upload validation."""
    valid_file = tmp_path / "test.tif"
    valid_file.write_bytes(b"test data")

    validate_upload_file(valid_file)

    invalid_file = tmp_path / "test.jpg"
    invalid_file.write_bytes(b"test data")

    with pytest.raises(ValueError, match="Only GeoTIFF files"):
        validate_upload_file(invalid_file)

    non_existent = tmp_path / "missing.tif"
    with pytest.raises(ValueError, match="File does not exist"):
        validate_upload_file(non_existent)


def test_get_project_file_manager():
    """Test factory function for file manager."""
    manager = get_project_file_manager("test-project")

    assert isinstance(manager, ProjectFileManager)
    assert manager.project_id == "test-project"
