import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import aioboto3
import aiofiles
import aiofiles.os
import aiofiles.tempfile
from aiobotocore.config import AioConfig
from botocore.exceptions import ClientError

from app.core.config import Settings, StorageConfig
from app.core.logging import get_logger
from app.core.secrets import get_secrets_manager

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

logger = get_logger(__name__)


class StorageBackend(Protocol):
    """Protocol for storage operations - implementations handle local/S3/etc."""

    async def upload(self, local_path: Path, key: str) -> str:
        """Upload file and return the storage key."""
        ...

    async def download(self, key: str, local_path: Path) -> None:
        """Download file to local path."""
        ...

    async def get_url(self, key: str) -> str:
        """Get presigned/accessible URL for file."""
        ...

    async def delete(self, key: str) -> None:
        """Delete file from storage."""
        ...

    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix."""
        ...

    async def file_exists(self, key: str) -> bool:
        """Check if file exists."""
        ...


class LocalStorage:
    """Local filesystem implementation of StorageBackend."""

    def __init__(self, storage_config: StorageConfig) -> None:
        """Initialize local storage backend."""
        self.base_dir = Path(storage_config.output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_mb = storage_config.max_file_size_mb

    async def upload(self, local_path: Path, key: str) -> str:
        """Copy file to storage directory and return the key."""
        target_path = self.base_dir / key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Stream the file in chunks to avoid high memory usage for large files.
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
        if not await aiofiles.os.path.exists(source_path):
            raise FileNotFoundError(f"File not found: {key}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Stream the file in chunks to avoid high memory usage.
        async with (
            aiofiles.open(source_path, "rb") as src,
            aiofiles.open(local_path, "wb") as dst,
        ):
            async for chunk in src:
                await dst.write(chunk)
        logger.info(f"Downloaded {source_path} to {local_path}")

    async def get_url(
        self,
        key: str,
    ) -> str:
        """Get file URL (file:// for local storage)"""
        try:
            file_path = self.base_dir / key
            if not await aiofiles.os.path.exists(file_path):
                logger.warning(f"File not found for URL generation: {key}")
                return key
            return file_path.as_uri()
        except Exception as e:
            logger.warning(f"Could not generate URL for {key}: {e}")
            return key

    async def delete(self, key: str) -> None:
        """Delete file from local storage."""
        file_path = self.base_dir / key
        try:
            await aiofiles.os.remove(file_path)
            logger.info(f"Deleted {file_path}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {key}")

    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix in local storage."""
        start_path = self.base_dir / prefix
        if not await aiofiles.os.path.exists(start_path):
            return []

        if await aiofiles.os.path.isfile(start_path):
            return [prefix]

        files = []
        entries = await aiofiles.os.scandir(start_path)
        for entry in entries:
            if entry.is_file():
                files.append(str(Path(entry.path).relative_to(self.base_dir)))
        return sorted(files)

    async def file_exists(self, key: str) -> bool:
        """Check if file exists in local storage."""
        return await aiofiles.os.path.exists(self.base_dir / key)


class SourceCoopStorage:
    """Source Coop S3-compatible storage with lazy credential loading."""

    def __init__(self, storage_config: StorageConfig) -> None:
        """Initialize Source Coop storage backend."""
        self.config = storage_config.source_coop
        self._initialized = False
        self.bucket_name: str = ""
        self._session: aioboto3.Session | None = None

    async def _lazy_init(self) -> None:
        """Load credentials and configure S3 client on first use."""
        if self._initialized:
            return

        if self.config.use_secrets_manager:
            logger.info(
                f"Load SourceCoop creds from Secrets Manager: {self.config.secret_name}"
            )
            secrets_manager = get_secrets_manager(self.config.secrets_manager_region)
            credentials = await secrets_manager.get_source_coop_credentials(
                self.config.secret_name
            )
            self._access_key_id = credentials["access_key_id"]
            self._secret_access_key = credentials["secret_access_key"]
        else:
            if not self.config.access_key_id or not self.config.secret_access_key:
                raise ValueError("Source Coop credentials not configured.")
            self._access_key_id = self.config.access_key_id
            self._secret_access_key = self.config.secret_access_key

        self.bucket_name = self.config.bucket_name
        self._region = self.config.region
        self._endpoint_url = self.config.endpoint_url

        self._session = aioboto3.Session()
        self._initialized = True
        logger.debug("Source Coop storage initialized successfully.")

    @contextlib.asynccontextmanager
    async def _get_s3_client(self) -> "AsyncGenerator[S3Client, None]":
        """Yield a configured S3 client for Source Coop."""
        await self._lazy_init()
        if not self._session:
            raise RuntimeError("Source Coop session not initialized.")

        # Configure checksum calculation for Source Coop compatibility
        source_coop_config = AioConfig(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )

        async with self._session.client(
            "s3",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            config=source_coop_config,
        ) as s3:
            yield s3

    def _get_storage_key(self, key: str) -> str:
        """Prepend the repository path to the storage key."""
        return (
            f"{self.config.repository_path}/{key}"
            if self.config.repository_path
            else key
        )

    def _strip_repository_path(self, storage_key: str) -> str:
        """Remove the repository path prefix from a storage key."""
        if self.config.repository_path:
            return storage_key.removeprefix(f"{self.config.repository_path}/")
        return storage_key

    async def upload(self, local_path: Path, key: str) -> str:
        """Upload file to Source Coop and return the key."""
        storage_key = self._get_storage_key(key)
        async with self._get_s3_client() as s3:
            try:
                await s3.upload_file(str(local_path), self.bucket_name, storage_key)
                logger.info(
                    f"Uploaded {local_path} to s3://{self.bucket_name}/{storage_key}"
                )
                return key
            except ClientError as e:
                logger.error(f"Failed to upload {local_path} to Source Coop: {e}")
                raise

    async def download(self, key: str, local_path: Path) -> None:
        """Download file from Source Coop to local path."""
        storage_key = self._get_storage_key(key)
        async with self._get_s3_client() as s3:
            try:
                await s3.download_file(self.bucket_name, storage_key, str(local_path))
                logger.info(
                    f"Downloaded s3://{self.bucket_name}/{storage_key} to {local_path}"
                )
            except ClientError as e:
                logger.error(f"Failed to download {key} from Source Coop: {e}")
                raise

    async def get_url(self, key: str) -> str:
        """Generate clean Source Coop URL for file access"""
        try:
            await self._lazy_init()  # Ensure we have bucket_name
            storage_key = self._get_storage_key(key)

            url = f"{self._endpoint_url}/{self.bucket_name}/{storage_key}"
            logger.debug(f"Generated Source Coop URL for {key}: {url}")
            return url
        except Exception as e:
            logger.warning(f"Could not generate URL for {key}: {e}")
            return key

    async def delete(self, key: str) -> None:
        """Delete file from Source Coop."""
        storage_key = self._get_storage_key(key)
        async with self._get_s3_client() as s3:
            try:
                await s3.delete_object(Bucket=self.bucket_name, Key=storage_key)
                logger.info(f"Deleted s3://{self.bucket_name}/{storage_key}")
            except ClientError as e:
                logger.error(f"Failed to delete {key} from Source Coop: {e}")
                raise

    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix in Source Coop."""
        storage_prefix = self._get_storage_key(prefix)
        async with self._get_s3_client() as s3:
            try:
                paginator = s3.get_paginator("list_objects_v2")
                keys: list[str] = []
                async for result in paginator.paginate(
                    Bucket=self.bucket_name, Prefix=storage_prefix
                ):
                    keys.extend(
                        content["Key"] for content in result.get("Contents", [])
                    )
                return [self._strip_repository_path(key) for key in keys]
            except ClientError as e:
                logger.error(f"Failed to list files with prefix {prefix}: {e}")
                raise

    async def file_exists(self, key: str) -> bool:
        """Check if file exists in Source Coop."""
        storage_key = self._get_storage_key(key)
        async with self._get_s3_client() as s3:
            try:
                await s3.head_object(Bucket=self.bucket_name, Key=storage_key)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False
                logger.error(f"Failed to check if {key} exists: {e}")
                raise


def get_storage(settings: Settings) -> StorageBackend:
    """Get storage backend based on the unified configuration."""
    storage_config = settings.storage

    if storage_config.backend == "source_coop":
        return SourceCoopStorage(storage_config)

    # Default to local storage
    return LocalStorage(storage_config)


@contextlib.asynccontextmanager
async def temp_files_context(*filenames: str) -> AsyncGenerator[list[Path], None]:
    """Context manager for temporary files with automatic cleanup."""
    async with aiofiles.tempfile.TemporaryDirectory() as temp_dir:
        temp_paths = [Path(temp_dir) / filename for filename in filenames]
        logger.debug(f"Created temp files: {temp_paths}")
        yield temp_paths
        logger.debug(f"Cleaning up temp directory: {temp_dir}")


def validate_upload_file(file_path: Path) -> None:
    """Validate uploaded file - only GeoTIFF files allowed."""
    if not file_path.exists():
        raise ValueError("File does not exist")
    if file_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Only GeoTIFF files (.tif, .tiff) are allowed")
