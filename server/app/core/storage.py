import contextlib
import shutil
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Protocol

import aioboto3
import aiofiles
from botocore.exceptions import ClientError

from app.core.config import S3Config, Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageBackend(Protocol):
    """Protocol for storage operations - implementations handle local/S3/etc."""

    async def upload(self, local_path: Path, key: str) -> str:
        """Upload file and return the storage key"""
        ...

    async def download(self, key: str, local_path: Path) -> None:
        """Download file to local path"""
        ...

    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Get presigned/accessible URL for file"""
        ...

    async def delete(self, key: str) -> None:
        """Delete file from storage"""
        ...

    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix"""
        ...

    async def file_exists(self, key: str) -> bool:
        """Check if file exists"""
        ...


class S3Storage:
    """S3 implementation of StorageBackend"""

    def __init__(self, s3_config: S3Config) -> None:
        """Initialize S3 storage backend."""
        if s3_config.bucket_name is None:
            raise ValueError("S3 bucket name is required but not configured")

        self.bucket_name: str = s3_config.bucket_name
        self.region = s3_config.region
        self.presigned_url_expiry = s3_config.presigned_url_expiry
        self.session = aioboto3.Session()

    async def upload(self, local_path: Path, key: str) -> str:
        """Upload file to S3 and return the S3 key."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                await s3.upload_file(str(local_path), self.bucket_name, key)
                logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{key}")
                return key
            except ClientError as e:
                logger.error(f"Failed to upload {local_path} to S3: {e}")
                raise

    async def download(self, key: str, local_path: Path) -> None:
        """Download file from S3 to local path."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                await s3.download_file(self.bucket_name, key, str(local_path))
                logger.info(f"Downloaded s3://{self.bucket_name}/{key} to {local_path}")
            except ClientError as e:
                logger.error(f"Failed to download {key} from S3: {e}")
                raise

    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for file access."""
        if expires_in <= 0:
            expires_in = self.presigned_url_expiry

        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": key},
                    ExpiresIn=expires_in,
                )
                logger.debug(f"Generated presigned URL for {key}")
                return str(url)
            except ClientError as e:
                logger.error(f"Failed to generate presigned URL for {key}: {e}")
                raise

    async def delete(self, key: str) -> None:
        """Delete file from S3."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                await s3.delete_object(Bucket=self.bucket_name, Key=key)
                logger.info(f"Deleted s3://{self.bucket_name}/{key}")
            except ClientError as e:
                logger.error(f"Failed to delete {key} from S3: {e}")
                raise

    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix in S3."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=prefix
                )
                if "Contents" not in response:
                    return []
                return [obj["Key"] for obj in response["Contents"]]
            except ClientError as e:
                logger.error(f"Failed to list files with prefix {prefix}: {e}")
                raise

    async def file_exists(self, key: str) -> bool:
        """Check if file exists in S3."""
        async with self.session.client("s3", region_name=self.region) as s3:
            try:
                await s3.head_object(Bucket=self.bucket_name, Key=key)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False
                logger.error(f"Failed to check if {key} exists: {e}")
                raise


class LocalStorage:
    """Local filesystem implementation of StorageBackend"""

    def __init__(self, base_dir: Path) -> None:
        """Initialize local storage backend."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def upload(self, local_path: Path, key: str) -> str:
        """Copy file to storage directory and return the key."""
        target_path = self.base_dir / key
        target_path.parent.mkdir(parents=True, exist_ok=True)

        async with (
            aiofiles.open(local_path, "rb") as src,
            aiofiles.open(target_path, "wb") as dst,
        ):
            async for chunk in src:
                await dst.write(chunk)
        logger.info(f"Copied {local_path} to {target_path}")

        return key

    async def download(self, key: str, local_path: Path) -> None:
        """Copy file from storage to local path."""
        source_path = self.base_dir / key
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {key}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with (
            aiofiles.open(source_path, "rb") as src,
            aiofiles.open(local_path, "wb") as dst,
        ):
            async for chunk in src:
                await dst.write(chunk)
        logger.info(f"Downloaded {source_path} to {local_path}")

    async def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Get file URL (file:// for local storage)."""
        file_path = self.base_dir / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {key}")
        return f"file://{file_path.absolute()}"

    async def delete(self, key: str) -> None:
        """Delete file from local storage."""
        file_path = self.base_dir / key
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted {file_path}")
        else:
            logger.warning(f"File not found for deletion: {key}")

    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix in local storage."""
        prefix_path = self.base_dir / prefix
        if not prefix_path.exists():
            return []

        files = []
        if prefix_path.is_dir():
            for file_path in prefix_path.rglob("*"):
                if file_path.is_file():
                    relative_key = str(file_path.relative_to(self.base_dir))
                    files.append(relative_key)
        elif prefix_path.is_file():
            files.append(prefix)

        return sorted(files)

    async def file_exists(self, key: str) -> bool:
        """Check if file exists in local storage."""
        file_path = self.base_dir / key
        return file_path.exists()


def get_storage(settings: Settings) -> StorageBackend:
    """Get storage backend based on configuration"""
    if settings.s3.enabled:
        return S3Storage(settings.s3)
    return LocalStorage(Path(settings.storage.output_dir))


@contextlib.asynccontextmanager
async def temp_file_context(filename: str) -> AsyncGenerator[Path, None]:
    """Context manager for temporary files with automatic cleanup."""
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / filename

    try:
        logger.debug(f"Created temp file: {temp_path}")
        yield temp_path
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")


@contextlib.asynccontextmanager
async def temp_files_context(*filenames: str) -> AsyncGenerator[list[Path], None]:
    """Context manager for multiple temporary files with automatic cleanup."""
    temp_dir = Path(tempfile.mkdtemp())
    temp_paths = [temp_dir / filename for filename in filenames]

    try:
        logger.debug(f"Created temp files: {temp_paths}")
        yield temp_paths
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")


def validate_upload_file(file_path: Path) -> None:
    """Validate uploaded file - only GeoTIFF files allowed."""
    if not file_path.exists():
        raise ValueError("File does not exist")

    if file_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Only GeoTIFF files (.tif) are allowed")
