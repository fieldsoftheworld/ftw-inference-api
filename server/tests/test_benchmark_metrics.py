"""Tests for FTW benchmark metrics (IoU matching and 0–100 scores)."""

from shapely.geometry import box

from app.benchmark.metrics import (
    ChipMetrics,
    Scores100,
    aggregate_to_scores_100,
    greedy_match_metrics,
)


def test_perfect_match_scores_100() -> None:
    """Identical single polygon: full recall, precision, IoU."""
    g = box(0, 0, 1, 1)
    cm = greedy_match_metrics([g], [g], iou_threshold=0.5)
    assert cm.gt_count == 1
    assert cm.pred_count == 1
    assert cm.matched_gt == 1
    s = aggregate_to_scores_100([cm])
    assert s.finding_fields == 100.0
    assert s.not_finding_non_fields == 100.0
    assert s.correct_sizes_and_shapes == 100.0


def test_missing_field_lowers_recall() -> None:
    """No predictions: precision undefined treated as 1.0, recall 0."""
    g = box(0, 0, 1, 1)
    cm = greedy_match_metrics([], [g], iou_threshold=0.5)
    s = aggregate_to_scores_100([cm])
    assert s.finding_fields == 0.0
    assert s.not_finding_non_fields == 100.0


def test_false_positive_lowers_precision() -> None:
    """Prediction with no GT: recall 0 (no GT labels), precision 0 (FP)."""
    p = box(0, 0, 1, 1)
    cm = greedy_match_metrics([p], [], iou_threshold=0.5)
    s = aggregate_to_scores_100([cm])
    assert s.finding_fields == 0.0
    assert s.not_finding_non_fields == 0.0  # FP


def test_aggregate_multiple_chips() -> None:
    chips = [
        ChipMetrics(2, 2, 2, 2, 0.8, 0.5),
        ChipMetrics(2, 2, 1, 1, 0.6, 0.5),
    ]
    s = aggregate_to_scores_100(chips)
    assert isinstance(s, Scores100)
