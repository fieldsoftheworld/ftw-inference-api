"""Behavioral tests for the /v1/feedback/* endpoints."""

import pytest
from app.db.models import FeedbackRecord
from fastapi.testclient import TestClient

TILE_RATING_URL = "/v1/feedback/rating"
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


def test_tile_rating_with_valid_good_tags(client: TestClient) -> None:
    """Tags valid for rating 3 (Good) are accepted."""
    payload = {
        **VALID_TILE_RATING,
        "rating": 3,
        "tags": ["clean_boundaries", "good_shapes"],
    }
    response = client.post(TILE_RATING_URL, json=payload)
    assert response.status_code == 200


def test_tile_rating_with_valid_poor_tags(client: TestClient) -> None:
    """Tags valid for rating 1 (Poor) are accepted."""
    payload = {
        **VALID_TILE_RATING,
        "rating": 1,
        "tags": ["fragmented", "missing_fields"],
    }
    response = client.post(TILE_RATING_URL, json=payload)
    assert response.status_code == 200


def test_tile_rating_wrong_tags_for_rating(client: TestClient) -> None:
    """Poor tags submitted with rating 3 are rejected with 400."""
    payload = {**VALID_TILE_RATING, "rating": 3, "tags": ["fragmented"]}
    response = client.post(TILE_RATING_URL, json=payload)
    assert response.status_code == 400


def test_tile_rating_duplicate_tags_rejected(client: TestClient) -> None:
    """Duplicate tag values are rejected with 400."""
    payload = {**VALID_TILE_RATING, "rating": 1, "tags": ["fragmented", "fragmented"]}
    response = client.post(TILE_RATING_URL, json=payload)
    assert response.status_code == 400


def test_tile_rating_acceptable_rating_accepts_poor_tags(client: TestClient) -> None:
    """Rating 2 (Acceptable) accepts the poor-tag set (validator else-branch)."""
    payload = {**VALID_TILE_RATING, "rating": 2, "tags": ["jagged_boundaries"]}
    response = client.post(TILE_RATING_URL, json=payload)
    assert response.status_code == 200


def test_tile_rating_empty_tags_list_rejected(client: TestClient) -> None:
    """An empty tags list is rejected (min_length=1 constraint)."""
    payload = {**VALID_TILE_RATING, "rating": 1, "tags": []}
    response = client.post(TILE_RATING_URL, json=payload)
    assert response.status_code == 400


def test_old_tile_rating_path_returns_404(client: TestClient) -> None:
    """The pre-rename /tile-rating path no longer exists (PR #80 rename)."""
    response = client.post("/v1/feedback/tile-rating", json=VALID_TILE_RATING)
    assert response.status_code == 404


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
    feedback_client.post(
        TILE_RATING_URL,
        json={**VALID_TILE_RATING, "rating": 3, "tags": ["clean_boundaries"]},
    )
    feedback_client.post(
        TILE_RATING_URL, json={**VALID_TILE_RATING, "rating": 1, "tags": ["fragmented"]}
    )

    response = feedback_client.get(AREA_SUMMARY_URL, params={"bbox": VALID_BBOX_STR})
    assert response.status_code == 200
    data = response.json()
    assert data["total_ratings"] == 2
    assert data["average_rating"] == pytest.approx(2.0)
    dist = {item["level"]: item["count"] for item in data["rating_distribution"]}
    assert dist[1] == 1
    assert dist[3] == 1
    assert "tag_counts" in data
    tag_map = {item["tag"]: item["count"] for item in data["tag_counts"]}
    assert tag_map["clean_boundaries"] == 1
    assert tag_map["fragmented"] == 1


def test_area_summary_tag_counts_empty_when_no_tags(
    feedback_client: TestClient,
) -> None:
    """area-summary returns tag_counts as empty list when ratings have no tags."""
    feedback_client.post(TILE_RATING_URL, json=VALID_TILE_RATING)

    response = feedback_client.get(AREA_SUMMARY_URL, params={"bbox": VALID_BBOX_STR})
    assert response.status_code == 200
    assert response.json()["tag_counts"] == []


def test_area_summary_tag_counts_sorted_descending(
    feedback_client: TestClient,
) -> None:
    """tag_counts is sorted by count descending — most frequent tag appears first."""
    # Seed: "fragmented" appears 3x, "missing_fields" appears 1x
    for _ in range(3):
        feedback_client.post(
            TILE_RATING_URL,
            json={**VALID_TILE_RATING, "rating": 1, "tags": ["fragmented"]},
        )
    feedback_client.post(
        TILE_RATING_URL,
        json={**VALID_TILE_RATING, "rating": 1, "tags": ["missing_fields"]},
    )

    response = feedback_client.get(AREA_SUMMARY_URL, params={"bbox": VALID_BBOX_STR})
    assert response.status_code == 200
    tag_counts = response.json()["tag_counts"]
    assert [item["tag"] for item in tag_counts] == ["fragmented", "missing_fields"]
    assert [item["count"] for item in tag_counts] == [3, 1]
