"""Fallback STAC item resolution via Planetary Computer HTTP API.

The ``ftw inference scene-selection`` subprocess sometimes gets HTTP 403 from STAC
endpoints (urllib user-agent / network policies). The public PC ``/search`` endpoint
accepts a normal User-Agent and returns STAC Item ``self`` links usable by FTW
inference the same way scene-selection does.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import requests

from app.core.logging import get_logger

logger = get_logger(__name__)

_PC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
_UA = "ftw-inference-api/0.1 (benchmark; +https://github.com/fieldsoftheworld)"


def pc_sentinel2_item_urls_for_bbox(
    bbox: list[float],
    year: int,
    *,
    need_two_temporal_windows: bool,
    timeout: int = 90,
) -> tuple[str | None, str | None]:
    """Return STAC Item self-hrefs for Sentinel-2 L2A over *bbox* in *year*.

    If *need_two_temporal_windows* is True, returns earliest and latest distinct
    items in the result set (by acquisition time) as a coarse stand-in for
    “start / end of season” windows when parquet has no STAC columns.
    """
    minx, miny, maxx, maxy = bbox
    dt = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
    body: dict[str, Any] = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [minx, miny, maxx, maxy],
        "datetime": dt,
        "limit": 50,
    }
    headers = {"User-Agent": _UA, "Content-Type": "application/json"}
    try:
        r = requests.post(_PC_SEARCH, json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
    except Exception as exc:
        logger.warning("PC STAC search failed", extra={"error": str(exc)})
        return None, None

    features = r.json().get("features") or []
    if not features:
        logger.warning(
            "PC STAC search returned no features",
            extra={"bbox": bbox, "year": year},
        )
        return None, None

    def _dt(feat: dict[str, Any]) -> str:
        p = feat.get("properties") or {}
        return str(p.get("datetime") or p.get("start_datetime") or "")

    def _mgrs_tile(feat: dict[str, Any]) -> str:
        p = feat.get("properties") or {}
        return str(p.get("s2:mgrs_tile") or p.get("grid:code") or "").strip()

    # ``ftw inference download`` requires both scenes' footprints to intersect. Picking
    # "earliest" and "latest" items globally often yields two different MGRS tiles
    # that do not intersect → silent download skip and zero benchmark chips.
    tile_counts = Counter(_mgrs_tile(f) for f in features if _mgrs_tile(f))
    if tile_counts:
        primary_tile, _ = tile_counts.most_common(1)[0]
        features = [f for f in features if _mgrs_tile(f) == primary_tile]

    sorted_feats = sorted(features, key=_dt)
    urls: list[str] = []
    for feat in sorted_feats:
        href = _self_href_from_feature(feat)
        if href:
            urls.append(href)
    seen: set[str] = set()
    uniq: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    if not uniq:
        return None, None

    if not need_two_temporal_windows:
        return uniq[0], None

    if len(uniq) >= 2:
        return uniq[0], uniq[-1]
    return uniq[0], uniq[0]


def _self_href_from_feature(feat: dict[str, Any]) -> str | None:
    for link in feat.get("links") or []:
        if link.get("rel") == "self" and link.get("href"):
            return str(link["href"]).strip()
    return None
