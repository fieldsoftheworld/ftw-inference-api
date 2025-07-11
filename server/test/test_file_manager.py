import pytest
import pytest_asyncio
from app.core.config import S3Config
from app.core.file_manager import (
    S3FileManager,
    get_project_file_manager,
    validate_upload_file,
)


@pytest_asyncio.fixture
async def s3_client_with_bucket(aioboto3_s3_client):
    """Create bucket in the mocked S3 environment."""
    async with aioboto3_s3_client as s3_client:
        await s3_client.create_bucket(Bucket="test-bucket")
        yield s3_client

        # Clean up bucket after test
        try:
            response = await s3_client.list_objects_v2(Bucket="test-bucket")
            if "Contents" in response:
                objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
                await s3_client.delete_objects(
                    Bucket="test-bucket", Delete={"Objects": objects}
                )
        except Exception:
            pass


@pytest_asyncio.fixture
async def project_manager(s3_client_with_bucket):
    """Create a project file manager with S3 configuration."""
    s3_config = S3Config(
        enabled=True,
        bucket_name="test-bucket",
        region="us-west-2",
        presigned_url_expiry=3600,
    )

    manager = S3FileManager("test-project-123", s3_config)
    yield manager


def test_project_file_manager_initialization():
    """Test S3FileManager initializes with correct configuration."""
    s3_config = S3Config(
        enabled=True,
        bucket_name="test-bucket",
        region="us-west-2",
        presigned_url_expiry=3600,
    )
    manager = S3FileManager("test-project", s3_config)

    assert manager.project_id == "test-project"
    assert manager.bucket_name == "test-bucket"
    assert manager.region == "us-west-2"
    assert "test-project" in manager.base_prefix


@pytest.mark.asyncio
async def test_ensure_directories(project_manager):
    """Test directory creation (S3 doesn't require this)."""
    await project_manager.ensure_directories()


@pytest.mark.asyncio
async def test_get_upload_path(project_manager):
    """Test upload path generation."""
    path_a = await project_manager.get_upload_path("a")
    path_b = await project_manager.get_upload_path("b")

    assert path_a.endswith("a.tif")
    assert path_b.endswith("b.tif")
    assert "test-project-123" in path_a
    assert "test-project-123" in path_b

    with pytest.raises(ValueError, match="Window must be 'a' or 'b'"):
        await project_manager.get_upload_path("c")


@pytest.mark.asyncio
async def test_get_result_path(project_manager):
    """Test result path generation."""
    inference_path = await project_manager.get_result_path("inference")
    polygon_path = await project_manager.get_result_path("polygons")

    assert inference_path.endswith(".tif")
    assert "inference" in inference_path
    assert polygon_path.endswith(".json")
    assert "polygons" in polygon_path
    assert "test-project-123" in inference_path
    assert "test-project-123" in polygon_path

    with pytest.raises(ValueError, match="Unknown result type"):
        await project_manager.get_result_path("unknown")


@pytest.mark.asyncio
async def test_get_temp_path(project_manager):
    """Test temporary path generation."""
    temp_path = project_manager.get_temp_path("test_file.txt")

    assert temp_path.name == "test_file.txt"
    assert temp_path.is_absolute()


@pytest.mark.asyncio
async def test_list_uploaded_images(project_manager, s3_client_with_bucket):
    """Test listing uploaded images."""
    s3_client = s3_client_with_bucket
    images = await project_manager.list_uploaded_images()
    assert len(images) == 0
    assert isinstance(images, dict)

    # Test with uploaded images
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/uploads/a.tif",
        Body=b"fake tif data",
    )
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/uploads/b.tif",
        Body=b"fake tif data",
    )

    images = await project_manager.list_uploaded_images()
    assert len(images) == 2
    assert "a" in images
    assert "b" in images


@pytest.mark.asyncio
async def test_has_uploaded_images(project_manager, s3_client_with_bucket):
    """Test checking for uploaded images."""
    s3_client = s3_client_with_bucket
    # Initially no images
    has_images = await project_manager.has_uploaded_images()
    assert has_images is False

    # Upload one image
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/uploads/a.tif",
        Body=b"fake tif data",
    )

    has_images = await project_manager.has_uploaded_images()
    assert has_images is False  # Need at least 2 images

    # Upload second image
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/uploads/b.tif",
        Body=b"fake tif data",
    )

    has_images = await project_manager.has_uploaded_images()
    assert has_images is True


@pytest.mark.asyncio
async def test_get_latest_inference_result(project_manager, s3_client_with_bucket):
    """Test getting latest inference result."""
    s3_client = s3_client_with_bucket

    result = await project_manager.get_latest_inference_result()
    assert result is None

    # Upload some inference results
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/results/older.inference.tif",
        Body=b"fake inference data",
    )
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/results/newer.inference.tif",
        Body=b"fake inference data",
    )

    result = await project_manager.get_latest_inference_result()
    assert result is not None
    assert result.endswith(".inference.tif")
    assert "test-project-123" in result


@pytest.mark.asyncio
async def test_cleanup_temp_files(project_manager):
    """Test cleanup of temporary files."""
    await project_manager.cleanup_temp_files()


@pytest.mark.asyncio
async def test_cleanup_all_files(project_manager, s3_client_with_bucket):
    """Test cleanup of all project files."""
    s3_client = s3_client_with_bucket

    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/uploads/a.tif",
        Body=b"fake tif data",
    )
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/results/test.inference.tif",
        Body=b"fake inference data",
    )

    response = await s3_client.list_objects_v2(
        Bucket="test-bucket", Prefix="projects/test-project-123/"
    )
    assert "Contents" in response
    assert len(response["Contents"]) == 2

    await project_manager.cleanup_all_files()

    # Verify files are gone
    response = await s3_client.list_objects_v2(
        Bucket="test-bucket", Prefix="projects/test-project-123/"
    )
    assert "Contents" not in response


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


@pytest.mark.asyncio
async def test_upload_and_download_file(
    project_manager, s3_client_with_bucket, tmp_path
):
    """Test file upload and download operations."""
    s3_client = s3_client_with_bucket
    test_file = tmp_path / "test_image.tif"
    test_data = b"fake geotiff data for testing"
    test_file.write_bytes(test_data)

    s3_key = "projects/test-project-123/uploads/test.tif"
    presigned_url = await project_manager.upload_file(test_file, s3_key)

    assert presigned_url is not None
    assert "test-bucket" in presigned_url

    response = await s3_client.head_object(Bucket="test-bucket", Key=s3_key)
    assert response["ContentLength"] == len(test_data)

    download_path = tmp_path / "downloaded_test.tif"
    await project_manager.download_file(s3_key, download_path)

    assert download_path.exists()
    assert download_path.read_bytes() == test_data


@pytest.mark.asyncio
async def test_get_presigned_url(project_manager, s3_client_with_bucket):
    """Test presigned URL generation."""
    s3_client = s3_client_with_bucket

    s3_key = "projects/test-project-123/test-file.tif"
    await s3_client.put_object(Bucket="test-bucket", Key=s3_key, Body=b"test data")

    url = await project_manager.get_presigned_url(s3_key)
    assert url is not None
    assert "test-bucket" in url
    assert s3_key in url

    url_custom = await project_manager.get_presigned_url(s3_key, expires_in=1800)
    assert url_custom is not None
    assert url_custom != url  # Should be different due to different expiry


@pytest.mark.asyncio
async def test_list_results(project_manager, s3_client_with_bucket):
    """Test listing results by type."""
    s3_client = s3_client_with_bucket
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/results/result1.inference.tif",
        Body=b"inference data 1",
    )
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/results/result2.inference.tif",
        Body=b"inference data 2",
    )

    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/results/polygons1.polygons.json",
        Body=b'{"type": "FeatureCollection"}',
    )

    inference_results = await project_manager.list_results("inference")
    assert len(inference_results) == 2
    assert all(key.endswith(".inference.tif") for key in inference_results)

    polygon_results = await project_manager.list_results("polygons")
    assert len(polygon_results) == 1
    assert polygon_results[0].endswith(".polygons.json")

    with pytest.raises(ValueError, match="Unknown result type"):
        await project_manager.list_results("invalid")


@pytest.mark.asyncio
async def test_get_project_results(project_manager, s3_client_with_bucket):
    """Test getting project results with presigned URLs."""
    s3_client = s3_client_with_bucket
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/results/latest.inference.tif",
        Body=b"latest inference data",
    )
    await s3_client.put_object(
        Bucket="test-bucket",
        Key="projects/test-project-123/results/latest.polygons.json",
        Body=b'{"type": "FeatureCollection"}',
    )

    results = await project_manager.get_project_results()

    assert "inference" in results
    assert "polygons" in results
    assert results["inference"] is not None
    assert results["polygons"] is not None
    assert "test-bucket" in results["inference"]
    assert "test-bucket" in results["polygons"]


def test_get_project_file_manager():
    """Test factory function for file manager."""
    manager = get_project_file_manager("test-project")

    assert isinstance(manager, S3FileManager)
    assert manager.project_id == "test-project"
