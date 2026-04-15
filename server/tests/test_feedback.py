"""Behavioral tests for the /v1/feedback/* endpoints."""

import pytest
from app.db.models import FeedbackRecord
from fastapi.testclient import TestClient

TILE_RATING_URL = "/v1/feedback/tile-rating"
TELL_US_MORE_URL = "/v1/feedback/tell-us-more"
CONTRIBUTE_URL = "/v1/feedback/contribute"
AREA_SUMMARY_URL = "/v1/feedback/area-summary"

VALID_BBOX = [12.0, 48.0, 13.0, 49.0]
VALID_BBOX_STR = "12.0,48.0,13.0,49.0"

VALID_TILE_RATING = {
    "rating": 2,
    "bbox": VALID_BBOX,
    "resolution": 76.4,
}

VALID_TELL_US_MORE = {
    "quality_feedback": "Boundaries look reasonable but miss some small parcels.",
    "use_case": "Crop monitoring for smallholder farms.",
    "bbox": VALID_BBOX,
    "resolution": 76.4,
}

VALID_CONTRIBUTE = {
    "contribution_types": ["annotator", "share_data"],
    "name": "Jane Doe",
    "email": "jane@example.com",
}


@pytest.fixture()
def feedback_client(client: TestClient) -> TestClient:
    """Test client with a clean FeedbackRecord table for area-summary isolation."""
    for item in FeedbackRecord.scan():
        item.delete()
    return client


# ---------------------------------------------------------------------------
# tile-rating
# ---------------------------------------------------------------------------


def test_tile_rating_success(client: TestClient) -> None:
    """A valid rating returns 200 with rating_id and status='created'."""
    response = client.post(TILE_RATING_URL, json=VALID_TILE_RATING)
    assert response.status_code == 200
    data = response.json()
    assert "rating_id" in data
    assert data["status"] == "created"


def test_tile_rating_invalid_rating_value(client: TestClient) -> None:
    """A rating outside [1, 2, 3] is rejected with 400 (app validation handler)."""
    payload = {**VALID_TILE_RATING, "rating": 4}
    response = client.post(TILE_RATING_URL, json=payload)
    assert response.status_code == 400


def test_tile_rating_invalid_bbox_order(client: TestClient) -> None:
    """A bbox where minLng > maxLng is rejected with 400 (app validation handler)."""
    payload = {**VALID_TILE_RATING, "bbox": [13.0, 48.0, 12.0, 49.0]}
    response = client.post(TILE_RATING_URL, json=payload)
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# tell-us-more
# ---------------------------------------------------------------------------


def test_tell_us_more_success(client: TestClient) -> None:
    """A valid submission returns 200 with feedback_id and status='submitted'."""
    response = client.post(TELL_US_MORE_URL, json=VALID_TELL_US_MORE)
    assert response.status_code == 200
    data = response.json()
    assert "feedback_id" in data
    assert data["status"] == "submitted"


def test_tell_us_more_invalid_email(client: TestClient) -> None:
    """An email lacking '@' is rejected with 400 (app validation handler)."""
    payload = {**VALID_TELL_US_MORE, "email": "not-an-email"}
    response = client.post(TELL_US_MORE_URL, json=payload)
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# contribute
# ---------------------------------------------------------------------------


def test_contribute_success(client: TestClient) -> None:
    """A valid submission returns 200 with contribution_id and submitted status."""
    response = client.post(CONTRIBUTE_URL, json=VALID_CONTRIBUTE)
    assert response.status_code == 200
    data = response.json()
    assert "contribution_id" in data
    assert data["status"] == "submitted"


def test_contribute_duplicate_types(client: TestClient) -> None:
    """Duplicate contribution_types values are rejected with 400."""
    payload = {**VALID_CONTRIBUTE, "contribution_types": ["annotator", "annotator"]}
    response = client.post(CONTRIBUTE_URL, json=payload)
    assert response.status_code == 400


def test_contribute_invalid_email(client: TestClient) -> None:
    """An email lacking '@' is rejected with 400 (app validation handler)."""
    payload = {**VALID_CONTRIBUTE, "email": "notvalid"}
    response = client.post(CONTRIBUTE_URL, json=payload)
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# area-summary
# ---------------------------------------------------------------------------


def test_area_summary_404_when_no_ratings(feedback_client: TestClient) -> None:
    """area-summary returns 404 when no ratings exist for the requested bbox."""
    response = feedback_client.get(AREA_SUMMARY_URL, params={"bbox": VALID_BBOX_STR})
    assert response.status_code == 404


def test_area_summary_returns_aggregated_ratings(feedback_client: TestClient) -> None:
    """After seeding two tile ratings, area-summary returns 200 with correct stats."""
    feedback_client.post(TILE_RATING_URL, json={**VALID_TILE_RATING, "rating": 3})
    feedback_client.post(TILE_RATING_URL, json={**VALID_TILE_RATING, "rating": 1})

    response = feedback_client.get(AREA_SUMMARY_URL, params={"bbox": VALID_BBOX_STR})
    assert response.status_code == 200
    data = response.json()
    assert data["total_ratings"] == 2
    assert data["average_rating"] == pytest.approx(2.0)
    dist = {item["level"]: item["count"] for item in data["rating_distribution"]}
    assert dist[1] == 1
    assert dist[3] == 1
