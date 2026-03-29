"""Instance polygon matching and 0–100 scores (recall, precision, geometry)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


def _as_poly(g: dict[str, Any] | BaseGeometry) -> BaseGeometry:
    if isinstance(g, BaseGeometry):
        return g
    return shape(g)


def polygon_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """Intersection over union for two polygons."""
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


@dataclass
class ChipMetrics:
    """Per-chip aggregated metrics before scaling to 0–100."""

    gt_count: int
    pred_count: int
    matched_gt: int
    matched_pred: int
    mean_iou_matched: float
    iou_threshold: float


@dataclass
class Scores100:
    """User-facing scores 0–100."""

    finding_fields: float
    not_finding_non_fields: float
    correct_sizes_and_shapes: float


def features_to_geometries(
    features: list[dict[str, Any]],
) -> list[BaseGeometry]:
    """Extract shapely geometries from GeoJSON features."""
    out: list[BaseGeometry] = []
    for f in features:
        geom = f.get("geometry")
        if not geom:
            continue
        try:
            g = _as_poly(geom)
            if not g.is_empty:
                out.append(g)
        except Exception:
            continue
    return out


def greedy_match_metrics(
    pred_geoms: list[BaseGeometry],
    gt_geoms: list[BaseGeometry],
    iou_threshold: float = 0.5,
) -> ChipMetrics:
    """Greedy IoU matching: assign each pred to best GT above threshold."""
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    ious: list[float] = []

    # Build IoU matrix
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred_geoms):
        for j, g in enumerate(gt_geoms):
            iou = polygon_iou(p, g)
            pairs.append((iou, i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)

    for iou, i, j in pairs:
        if iou < iou_threshold:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        ious.append(iou)

    matched = len(ious)
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return ChipMetrics(
        gt_count=len(gt_geoms),
        pred_count=len(pred_geoms),
        matched_gt=matched,
        matched_pred=matched,
        mean_iou_matched=mean_iou,
        iou_threshold=iou_threshold,
    )


def aggregate_to_scores_100(
    chips: list[ChipMetrics],
) -> Scores100:
    """Micro-average chip metrics into three 0–100 scores."""
    if not chips:
        return Scores100(0.0, 0.0, 0.0)

    total_gt = sum(c.gt_count for c in chips)
    total_pred = sum(c.pred_count for c in chips)
    total_matched = sum(c.matched_gt for c in chips)
    sum_iou = sum(c.mean_iou_matched * c.matched_gt for c in chips if c.matched_gt > 0)
    matched_for_iou = sum(c.matched_gt for c in chips)

    # 1. Recall: finding all fields (no GT labels → 0, not 100 — avoids fake “perfect” recall)
    recall = (total_matched / total_gt) if total_gt > 0 else 0.0
    finding_fields = max(0.0, min(100.0, 100.0 * recall))

    # 2. Precision: not hallucinating fields (non-fields / FP)
    precision = (total_matched / total_pred) if total_pred > 0 else 1.0
    not_finding_non_fields = max(0.0, min(100.0, 100.0 * precision))

    # 3. Mean IoU on matched pairs (geometry / size proxy)
    mean_iou = (sum_iou / matched_for_iou) if matched_for_iou > 0 else 0.0
    correct_sizes = max(0.0, min(100.0, 100.0 * mean_iou))

    return Scores100(
        finding_fields=finding_fields,
        not_finding_non_fields=not_finding_non_fields,
        correct_sizes_and_shapes=correct_sizes,
    )


def geojson_dict_to_features(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize ftw GeoJSON FeatureCollection or dict with features."""
    if isinstance(data, dict) and "features" in data:
        return list(data.get("features") or [])
    return []
