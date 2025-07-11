import tempfile
import uuid
from pathlib import Path

import aioboto3
from botocore.exceptions import ClientError

from app.core.config import S3Config, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class S3FileManager:
    """Manages file operations for projects using S3 storage."""

    def __init__(self, project_id: str, s3_config: S3Config) -> None:
        """Initialize S3 file manager for a specific project."""
        self.project_id = project_id
        self.bucket_name = s3_config.bucket_name
        self.region = s3_config.region
        self.presigned_url_expiry = s3_config.presigned_url_expiry
        self.base_prefix = f"projects/{project_id}/"
        self.session = aioboto3.Session()

    async def ensure_directories(self) -> None:
        """S3 doesn't require directory creation."""
        pass

    async def get_upload_path(self, window: str) -> str:
        """Get the S3 key for a window image."""
        if window not in ["a", "b"]:
            raise ValueError("Window must be 'a' or 'b'")
        return f"{self.base_prefix}uploads/{window}.tif"

    async def get_result_path(self, result_type: str) -> str:
        """Get S3 key for storing results."""
        uid = str(uuid.uuid4())
        if result_type == "inference":
            return f"{self.base_prefix}results/{uid}.inference.tif"
        elif result_type == "polygons":
            return f"{self.base_prefix}results/{uid}.polygons.json"
        else:
            raise ValueError(f"Unknown result type: {result_type}")

    @staticmethod
    def get_temp_path(filename: str) -> Path:
        """Get temporary file path using system temp directory."""
        temp_dir = Path(tempfile.mkdtemp())
        return temp_dir / filename

    async def upload_file(self, local_path: Path, s3_key: str) -> str:
        """Upload file to S3 and return presigned URL for access."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                await s3.upload_file(str(local_path), self.bucket_name, s3_key)
                logger.info(
                    f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}"
                )
                return await self.get_presigned_url(s3_key)
            except ClientError as e:
                logger.error(f"Failed to upload {local_path} to S3: {e}")
                raise

    async def download_file(self, s3_key: str, local_path: Path) -> None:
        """Download file from S3 to local path."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                await s3.download_file(self.bucket_name, s3_key, str(local_path))
                logger.info(
                    f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}"
                )
            except ClientError as e:
                logger.error(f"Failed to download {s3_key} from S3: {e}")
                raise

    async def get_presigned_url(
        self, s3_key: str, expires_in: int | None = None
    ) -> str:
        """Generate presigned URL for file access."""
        if expires_in is None:
            expires_in = self.presigned_url_expiry

        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": s3_key},
                    ExpiresIn=expires_in,
                )
                logger.debug(f"Generated presigned URL for {s3_key}")
                return url
            except ClientError as e:
                logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
                raise

    async def list_uploaded_images(self) -> dict[str, str]:
        """List uploaded images for this project."""
        images = {}
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                for window in ["a", "b"]:
                    s3_key = f"{self.base_prefix}uploads/{window}.tif"
                    try:
                        await s3.head_object(Bucket=self.bucket_name, Key=s3_key)
                        images[window] = s3_key
                    except ClientError as e:
                        if e.response["Error"]["Code"] != "404":
                            raise
            except ClientError as e:
                logger.error(f"Failed to list uploaded images: {e}")
                raise
        return images

    async def has_uploaded_images(self) -> bool:
        """Check if project has uploaded images."""
        images = await self.list_uploaded_images()
        return len(images) >= 2

    async def get_latest_inference_result(self) -> str | None:
        """Get the most recent inference result S3 key."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"{self.base_prefix}results/",
                    Delimiter="/",
                )

                if "Contents" not in response:
                    return None

                inference_files = [
                    obj
                    for obj in response["Contents"]
                    if obj["Key"].endswith(".inference.tif")
                ]

                if not inference_files:
                    return None

                latest_file = max(inference_files, key=lambda f: f["LastModified"])
                return latest_file["Key"]
            except ClientError as e:
                logger.error(f"Failed to get latest inference result: {e}")
                raise

    async def get_latest_polygon_result(self) -> str | None:
        """Get the most recent polygon result S3 key."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"{self.base_prefix}results/",
                    Delimiter="/",
                )

                if "Contents" not in response:
                    return None

                polygon_files = [
                    obj
                    for obj in response["Contents"]
                    if obj["Key"].endswith(".polygons.json")
                ]

                if not polygon_files:
                    return None

                latest_file = max(polygon_files, key=lambda f: f["LastModified"])
                return latest_file["Key"]
            except ClientError as e:
                logger.error(f"Failed to get latest polygon result: {e}")
                raise

    async def list_results(self, result_type: str) -> list[str]:
        """List all results of given type."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"{self.base_prefix}results/",
                    Delimiter="/",
                )

                if "Contents" not in response:
                    return []

                if result_type == "inference":
                    suffix = ".inference.tif"
                elif result_type == "polygons":
                    suffix = ".polygons.json"
                else:
                    raise ValueError(f"Unknown result type: {result_type}")

                result_files = [
                    obj["Key"]
                    for obj in response["Contents"]
                    if obj["Key"].endswith(suffix)
                ]

                return sorted(result_files, key=lambda k: k.split("/")[-1])
            except ClientError as e:
                logger.error(f"Failed to list {result_type} results: {e}")
                raise

    async def cleanup_temp_files(self) -> None:
        """Cleanup is handled by system temp directory cleanup."""
        logger.info(
            f"Temp files cleanup handled by system for project {self.project_id}"
        )

    async def get_project_results(self) -> dict[str, str | None]:
        """Get URLs for latest inference and polygon results."""
        results: dict[str, str | None] = {"inference": None, "polygons": None}

        try:
            polygon_key = await self.get_latest_polygon_result()
            if polygon_key:
                results["polygons"] = await self.get_presigned_url(polygon_key)

            inference_key = await self.get_latest_inference_result()
            if inference_key:
                results["inference"] = await self.get_presigned_url(inference_key)
        except (ClientError, FileNotFoundError):
            pass

        return results

    async def cleanup_all_files(self) -> None:
        """Remove all files for this project from S3."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=self.base_prefix,
                )

                if "Contents" not in response:
                    logger.info(f"No files to cleanup for project {self.project_id}")
                    return

                objects_to_delete = [
                    {"Key": obj["Key"]} for obj in response["Contents"]
                ]

                if objects_to_delete:
                    await s3.delete_objects(
                        Bucket=self.bucket_name, Delete={"Objects": objects_to_delete}
                    )
                    logger.info(
                        f"Cleaned up {len(objects_to_delete)} files for "
                        f"project {self.project_id}"
                    )
            except ClientError as e:
                logger.error(
                    f"Failed to cleanup files for project {self.project_id}: {e}"
                )
                raise


def validate_upload_file(file_path: Path) -> None:
    """Validate uploaded file - only GeoTIFF files allowed."""
    if not file_path.exists():
        raise ValueError("File does not exist")

    if file_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Only GeoTIFF files (.tif) are allowed")


def get_project_file_manager(project_id: str) -> S3FileManager:
    """Get a file manager instance for a project."""
    # Keeping factory pattern for when we have source coop storage resolved
    settings = get_settings()
    return S3FileManager(project_id, settings.s3)
