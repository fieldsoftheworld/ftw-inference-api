import shutil
import uuid
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# Base directories
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")
TEMP_DIR = Path("data/temp")


class ProjectFileManager:
    """Manages file operations for projects."""

    def __init__(self, project_id: str) -> None:
        """Initialize file manager for a specific project."""
        self.project_id = project_id
        self.upload_dir = UPLOAD_DIR / project_id
        self.results_dir = RESULTS_DIR / project_id
        self.temp_dir = TEMP_DIR / project_id

    def ensure_directories(self) -> None:
        """Create project directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_upload_path(self, window: str) -> Path:
        """Get the upload path for a window image."""
        if window not in ["a", "b"]:
            raise ValueError("Window must be 'a' or 'b'")
        return self.upload_dir / f"{window}.tif"

    def get_result_path(self, result_type: str) -> Path:
        """Get path for storing results."""
        uid = str(uuid.uuid4())
        if result_type == "inference":
            return self.results_dir / f"{uid}.inference.tif"
        elif result_type == "polygons":
            return self.results_dir / f"{uid}.polygons.json"
        else:
            raise ValueError(f"Unknown result type: {result_type}")

    def get_temp_path(self, filename: str) -> Path:
        """Get temporary file path."""
        return self.temp_dir / filename

    def list_uploaded_images(self) -> dict[str, Path]:
        """List uploaded images for this project."""
        images = {}
        for window in ["a", "b"]:
            path = self.upload_dir / f"{window}.tif"
            if path.exists():
                images[window] = path
        return images

    def has_uploaded_images(self) -> bool:
        """Check if project has uploaded images."""
        return len(self.list_uploaded_images()) >= 2

    def get_latest_inference_result(self) -> Path | None:
        """Get the most recent inference result file."""
        if not self.results_dir.exists():
            return None
        inference_files = list(self.results_dir.glob("*.inference.tif"))
        if not inference_files:
            return None
        return max(inference_files, key=lambda f: f.stat().st_mtime)

    def cleanup_temp_files(self) -> None:
        """Remove temporary files for this project."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp files for project {self.project_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup temp files for {self.project_id}: {e}"
                )

    def cleanup_all_files(self) -> None:
        """Remove all files for this project."""
        for directory in [self.upload_dir, self.results_dir, self.temp_dir]:
            if directory.exists():
                try:
                    shutil.rmtree(directory)
                    logger.info(f"Cleaned up {directory} for project {self.project_id}")
                except Exception as e:
                    logger.warning(
                        f"Failed to cleanup {directory} for {self.project_id}: {e}"
                    )


def validate_upload_file(file_path: Path) -> None:
    """Validate uploaded file - only GeoTIFF files allowed."""
    if not file_path.exists():
        raise ValueError("File does not exist")

    if file_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Only GeoTIFF files (.tif) are allowed")


def get_project_file_manager(project_id: str) -> ProjectFileManager:
    """Get a file manager instance for a project."""
    return ProjectFileManager(project_id)
