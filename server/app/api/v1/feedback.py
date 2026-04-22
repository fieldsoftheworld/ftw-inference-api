import json
from collections import Counter

from fastapi import APIRouter, HTTPException, status
from pynamodb.exceptions import PutError, ScanError

from app.api.v1.dependencies import AuthDep, BBoxQueryDep
from app.core.logging import get_logger
from app.db.models import FeedbackRecord
from app.schemas.feedback import (
    AreaSummaryResponse,
    ContributeRequest,
    ContributeResponse,
    TellUsMoreRequest,
    TellUsMoreResponse,
    TileRatingRequest,
    TileRatingResponse,
)

logger = get_logger(__name__)

feedback_router = APIRouter(prefix="/feedback", tags=["Feedback"])


def _bboxes_intersect(a: list[float], b: list[float]) -> bool:
    """Return True if two WGS84 bboxes intersect."""
    a_min_lng, a_min_lat, a_max_lng, a_max_lat = a
    b_min_lng, b_min_lat, b_max_lng, b_max_lat = b
    return (
        a_min_lng <= b_max_lng
        and a_max_lng >= b_min_lng
        and a_min_lat <= b_max_lat
        and a_max_lat >= b_min_lat
    )


@feedback_router.post(
    "/rating",
    status_code=status.HTTP_200_OK,
)
async def submit_tile_rating(
    body: TileRatingRequest,
    auth: AuthDep,
) -> TileRatingResponse:
    """Submit a 1-3 quality rating for the field boundaries visible in the current
    viewport.

    **Location:** Identified by the WGS84 viewport bbox plus zoom level.

    **Deduplication:** If the same client submits multiple ratings for substantially
    the same bbox + zoom within a short window, the server treats the latest as an
    update. Response status will be `updated` in that case.

    **Rate limit:** 30 requests per minute per IP.
    """
    record = FeedbackRecord(
        feedback_type="tile_rating",
        bbox=json.dumps(body.bbox),
        resolution=body.resolution,
        payload=json.dumps(body.model_dump()),
    )
    try:
        record.save()
    except PutError:
        logger.error(
            "Failed to save tile_rating record",
            exc_info=True,
            extra={"feedback_type": "tile_rating"},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback could not be saved. Please try again.",
        ) from None
    logger.info("tile_rating saved", extra={"rating_id": record.id})
    # TODO: implement IP-based deduplication — if same IP submits for the same
    # bbox+zoom within a short window, update the existing record and return
    # status="updated" instead of creating a new one.
    return TileRatingResponse(rating_id=record.id, status="created")


@feedback_router.post(
    "/tell-us-more",
    status_code=status.HTTP_200_OK,
)
async def submit_tell_us_more(
    body: TellUsMoreRequest,
    auth: AuthDep,
) -> TellUsMoreResponse:
    """Submit detailed feedback about field boundary quality.

    Typically reached via a "Tell Us More" button after a quick tile rating.

    **Rate limit:** 5 requests per minute per IP.

    **Email notification:** Each successful submission will trigger an email
    notification to project maintainers (SNS integration pending).
    """
    record = FeedbackRecord(
        feedback_type="tell_us_more",
        bbox=json.dumps(body.bbox),
        resolution=body.resolution,
        payload=json.dumps(body.model_dump()),
    )
    try:
        record.save()
    except PutError:
        logger.error(
            "Failed to save tell_us_more record",
            exc_info=True,
            extra={"feedback_type": "tell_us_more"},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback could not be saved. Please try again.",
        ) from None
    logger.info("tell_us_more feedback saved", extra={"feedback_id": record.id})
    # TODO: trigger SNS notification to project maintainers
    return TellUsMoreResponse(feedback_id=record.id, status="submitted")


@feedback_router.post(
    "/contribute",
    status_code=status.HTTP_200_OK,
)
async def submit_contribute(
    body: ContributeRequest,
    auth: AuthDep,
) -> ContributeResponse:
    """Submit a community contribution interest form.

    Captures interest from annotators, data providers, model developers, and
    code contributors.

    **Rate limit:** 5 requests per minute per IP.

    **Email notification:** Each successful submission will trigger an email
    notification to project maintainers (SNS integration pending).
    """
    record = FeedbackRecord(
        feedback_type="contribute",
        payload=json.dumps(body.model_dump()),
    )
    try:
        record.save()
    except PutError:
        logger.error(
            "Failed to save contribute record",
            exc_info=True,
            extra={"feedback_type": "contribute"},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback could not be saved. Please try again.",
        ) from None
    logger.info("contribution form saved", extra={"contribution_id": record.id})
    # TODO: trigger SNS notification to project maintainers
    return ContributeResponse(contribution_id=record.id, status="submitted")


@feedback_router.get(
    "/area-summary",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"description": "No ratings found in this area"},
    },
)
async def get_area_summary(
    auth: AuthDep,
    bbox: BBoxQueryDep,
) -> AreaSummaryResponse:
    """Return aggregate rating statistics for all tile ratings whose viewport
    bbox intersects the requested bbox.

    Used by the frontend to highlight regions where users have provided feedback.

    **Rate limit:** 60 requests per minute per IP.
    """
    matching_ratings: list[int] = []
    matching_tags: list[str] = []

    try:
        scan_results = list(
            FeedbackRecord.scan(FeedbackRecord.feedback_type == "tile_rating")
        )
    except ScanError:
        logger.error("DynamoDB scan failed in get_area_summary", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Area summary is temporarily unavailable.",
        ) from None

    for record in scan_results:
        if not record.bbox:
            continue
        try:
            record_bbox: list[float] = json.loads(record.bbox)
            record_payload: dict = json.loads(record.payload)
        except (json.JSONDecodeError, ValueError):
            continue

        if _bboxes_intersect(bbox, record_bbox):
            rating = record_payload.get("rating")
            if isinstance(rating, int) and rating in (1, 2, 3):
                matching_ratings.append(rating)
            tags = record_payload.get("tags")
            if isinstance(tags, list):
                matching_tags.extend(tags)

    if not matching_ratings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No ratings found in this area",
        )

    total = len(matching_ratings)
    avg = round(sum(matching_ratings) / total, 2)
    tag_counter = Counter(matching_tags)
    tag_counts: list[dict[str, str | int]] = [
        {"tag": tag, "count": count}
        for tag, count in sorted(tag_counter.items(), key=lambda x: -x[1])
    ]

    return AreaSummaryResponse(
        bbox=bbox,
        total_ratings=total,
        average_rating=avg,
        rating_distribution=[
            {"level": 1, "count": matching_ratings.count(1)},
            {"level": 2, "count": matching_ratings.count(2)},
            {"level": 3, "count": matching_ratings.count(3)},
        ],
        tag_counts=tag_counts,
    )
