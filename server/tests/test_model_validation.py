import pytest
from app.schemas.requests import InferenceRequest
from fastapi.testclient import TestClient
from pydantic import ValidationError


class TestModelValidation:
    """Test model validation including single-window support."""

    def test_single_window_model_rejects_two_images(
        self, model_ids, sample_image_urls, sample_bbox
    ):
        """Test that single-window models reject 2 images."""
        with pytest.raises(ValueError, match="requires exactly 1 image"):
            InferenceRequest(
                model=model_ids["single_window"],
                images=sample_image_urls["dual"],  # Wrong: should be 1
                bbox=sample_bbox,
            )

    def test_single_window_model_accepts_one_image(
        self, model_ids, sample_image_urls, sample_bbox
    ):
        """Test that single-window models accept 1 image."""
        # Should not raise error
        request = InferenceRequest(
            model=model_ids["single_window"],
            images=sample_image_urls["single"],
            bbox=sample_bbox,
        )
        assert request.model == model_ids["single_window"]
        assert len(request.images) == 1

    def test_dual_window_model_rejects_one_image(
        self, model_ids, sample_image_urls, sample_bbox
    ):
        """Test that dual-window models reject 1 image."""
        with pytest.raises(ValueError, match="requires exactly 2 image"):
            InferenceRequest(
                model=model_ids["dual_window"],
                images=sample_image_urls["single"],  # Wrong: should be 2
                bbox=sample_bbox,
            )

    def test_dual_window_model_accepts_two_images(
        self, model_ids, sample_image_urls, sample_bbox
    ):
        """Test that dual-window models accept 2 images."""
        # Should not raise error
        request = InferenceRequest(
            model=model_ids["dual_window"],
            images=sample_image_urls["dual"],
            bbox=sample_bbox,
        )
        assert request.model == model_ids["dual_window"]
        assert len(request.images) == 2

    def test_validation_skipped_when_no_images_provided(self, model_ids, sample_bbox):
        """Test that image count validation is skipped when no images provided."""
        # For file-based workflows where images are uploaded separately
        request = InferenceRequest(
            model=model_ids["dual_window"],
            images=None,
            bbox=sample_bbox,
        )
        assert request.model == model_ids["dual_window"]
        assert request.images is None

    def test_unknown_model_rejected(self, sample_image_urls, sample_bbox):
        """Test that unknown model IDs are rejected with helpful message."""
        with pytest.raises(ValidationError, match="Input should be"):
            InferenceRequest(
                model="this-model-does-not-exist",
                images=sample_image_urls["single"],
                bbox=sample_bbox,
            )


class TestModelEndpoints:
    """Test model discovery endpoints."""

    def test_list_models_endpoint(self, client: TestClient):
        """Test /models endpoint returns all models."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "total" in data
        assert len(data["models"]) > 0

        # Check model structure
        model = data["models"][0]
        assert "id" in model
        assert "description" in model
        assert "license" in model
        assert "version" in model
        assert "requires_window" in model
        assert "requires_polygonize" in model
        assert "image_count" in model

    @pytest.mark.parametrize(
        "model_id,expected_requires_window,expected_image_count",
        [
            ("3_Class_FULL_singleWindow_v2", False, 1),
            ("2_Class_FULL_v1", True, 2),
        ],
    )
    def test_get_model_endpoint(
        self,
        client: TestClient,
        model_id: str,
        expected_requires_window: bool,
        expected_image_count: int,
    ):
        """Test /models/{id} endpoint returns correct model details."""
        response = client.get(f"/v1/models/{model_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == model_id
        assert data["requires_window"] is expected_requires_window
        assert data["image_count"] == expected_image_count
        assert "usage_example" in data
        assert len(data["usage_example"]["inference"]["images"]) == expected_image_count

    def test_get_nonexistent_model_returns_404(self, client: TestClient):
        """Test /models/{id} returns 404 for unknown model."""
        response = client.get("/v1/models/nonexistent-model-xyz")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
        assert "GET /v1/models" in data["detail"]


class TestModelIntegration:
    """Integration tests for model validation in workflows."""

    @pytest.mark.asyncio
    async def test_single_window_model_workflow(
        self,
        client: TestClient,
        dynamodb_tables,
        create_test_project,
        model_ids,
        sample_image_urls,
    ):
        """Test complete workflow with single-window model."""
        # Create project
        project_id = create_test_project("Single Window Test")

        # Submit inference with single-window model
        inference_params = {
            "model": model_ids["single_window"],
            "images": sample_image_urls["single"],
        }
        submit_response = client.put(
            f"/v1/projects/{project_id}/inference", json=inference_params
        )
        assert submit_response.status_code == 202
        assert "task_id" in submit_response.json()
