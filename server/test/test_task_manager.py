import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from app.core.task_manager import TaskManager, get_task_manager
from app.core.types import TaskStatus


@pytest_asyncio.fixture
async def task_manager():
    """Create a fresh TaskManager instance for testing."""
    manager = TaskManager(max_workers=1)
    await manager.start_workers()
    yield manager
    await manager.stop_workers()


@pytest.mark.asyncio
async def test_task_manager_initialization():
    """Test TaskManager initializes correctly."""
    manager = TaskManager(max_workers=2)
    assert manager.max_workers == 2
    assert len(manager.workers) == 0
    assert len(manager.active_tasks) == 0
    assert not manager.shutdown_event.is_set()


@pytest.mark.asyncio
async def test_submit_task(task_manager):
    """Test task submission returns task ID and tracks task."""
    task_data = {
        "task_type": "inference",
        "project_id": "test-project-123",
        "inference_params": {"model": "test-model"},
    }

    task_id = await task_manager.submit_task(task_data)

    assert task_id is not None
    assert len(task_id) > 0
    assert task_id in task_manager.active_tasks
    assert task_manager.active_tasks[task_id]["status"] == TaskStatus.PENDING.value
    assert task_manager.active_tasks[task_id]["project_id"] == "test-project-123"


@pytest.mark.asyncio
async def test_get_task_status(task_manager):
    """Test task status retrieval."""
    task_data = {"task_type": "inference", "project_id": "test-project"}
    task_id = await task_manager.submit_task(task_data)

    status = await task_manager.get_task_status(task_id)
    assert status == TaskStatus.PENDING

    status = await task_manager.get_task_status("non-existent")
    assert status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_get_task_info(task_manager):
    """Test task info retrieval."""
    task_data = {"task_type": "polygonize", "project_id": "test-project"}
    task_id = await task_manager.submit_task(task_data)

    info = await task_manager.get_task_info(task_id)
    assert info is not None
    assert info["status"] == TaskStatus.PENDING.value
    assert info["task_type"] == "polygonize"
    assert info["project_id"] == "test-project"

    info = await task_manager.get_task_info("non-existent")
    assert info is None


@pytest.mark.asyncio
async def test_task_processing_success():
    """Test successful task processing updates status correctly."""
    manager = TaskManager(max_workers=1)

    with patch.object(
        manager, "_process_inference_task", new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = {"result": "success"}

        await manager.start_workers()

        task_data = {
            "task_type": "inference",
            "project_id": "test-project",
            "inference_params": {"model": "test"},
        }
        task_id = await manager.submit_task(task_data)

        await asyncio.sleep(0.1)

        assert task_id in manager.active_tasks
        task_info = manager.active_tasks[task_id]
        assert task_info["status"] == TaskStatus.COMPLETED.value
        assert task_info["result"] == {"result": "success"}
        assert task_info["error"] is None

        await manager.stop_workers()


@pytest.mark.asyncio
async def test_task_processing_failure():
    """Test failed task processing updates status correctly."""
    manager = TaskManager(max_workers=1)

    with patch.object(
        manager, "_process_inference_task", new_callable=AsyncMock
    ) as mock_process:
        mock_process.side_effect = Exception("Processing failed")

        await manager.start_workers()

        task_data = {
            "task_type": "inference",
            "project_id": "test-project",
            "inference_params": {"model": "test"},
        }
        task_id = await manager.submit_task(task_data)

        await asyncio.sleep(0.1)

        assert task_id in manager.active_tasks
        task_info = manager.active_tasks[task_id]
        assert task_info["status"] == TaskStatus.FAILED.value
        assert "Processing failed" in task_info["error"]
        assert task_info["result"] is None

        await manager.stop_workers()


@pytest.mark.asyncio
async def test_unknown_task_type():
    """Test handling of unknown task types."""
    manager = TaskManager(max_workers=1)
    await manager.start_workers()

    task_data = {"task_type": "unknown", "project_id": "test-project"}
    task_id = await manager.submit_task(task_data)

    await asyncio.sleep(0.1)

    task_info = manager.active_tasks[task_id]
    assert task_info["status"] == TaskStatus.FAILED.value
    assert "Unknown task type" in task_info["error"]

    await manager.stop_workers()


@pytest.mark.asyncio
async def test_cleanup_completed_tasks(task_manager):
    """Test cleanup of old completed tasks."""
    task_data = {"task_type": "inference", "project_id": "test"}
    task_id = await task_manager.submit_task(task_data)

    task_manager.active_tasks[task_id]["status"] = TaskStatus.COMPLETED.value
    task_manager.active_tasks[task_id]["completed_at"] = (
        asyncio.get_event_loop().time() - 7200
    )

    await task_manager.cleanup_completed_tasks(max_age_seconds=3600)

    assert task_id not in task_manager.active_tasks


@pytest.mark.asyncio
async def test_worker_lifecycle():
    """Test worker start and stop lifecycle."""
    manager = TaskManager(max_workers=2)

    await manager.start_workers()
    assert len(manager.workers) == 2
    assert not manager.shutdown_event.is_set()

    await manager.stop_workers()
    assert len(manager.workers) == 0
    assert manager.shutdown_event.is_set()


def test_get_task_manager_singleton():
    """Test get_task_manager returns singleton instance."""
    manager1 = get_task_manager()
    manager2 = get_task_manager()
    assert manager1 is manager2
