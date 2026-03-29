"""Benchmark evaluation pipeline: chips parquet, GT masks, inference via InferenceService."""

from __future__ import annotations

import asyncio
import json
import math
import random
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from ftw_tools.inference.model_registry import MODEL_REGISTRY
from shapely.geometry import mapping as shp_mapping
from shapely.geometry import shape

from app.benchmark.download import (
    ChipsParquetNotFoundError,
    ensure_chips_parquet,
    ensure_data_config,
    ensure_label_masks,
)
from app.benchmark.manifest import get_country_by_id
from app.benchmark.metrics import (
    ChipMetrics,
    aggregate_to_scores_100,
    features_to_geometries,
    geojson_dict_to_features,
    greedy_match_metrics,
)
from app.benchmark.stac_search import pc_sentinel2_item_urls_for_bbox
from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging import get_logger
from app.services.inference_service import InferenceService

logger = get_logger(__name__)

# Stable cache when auto_download=True and no data_root/cache_dir (not cwd-relative).
_DEFAULT_CACHE = PROJECT_ROOT / "data" / "ftw_cache"

# GeoJSON map payloads scale with chip count; skip above this to keep responses bounded.
_MAP_GEOJSON_MAX_CHIPS = 40


def _json_safe(obj: Any) -> Any:
    """Make GeoJSON-like dicts JSON-serializable (numpy coords from Shapely, etc.)."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return obj


def _geoms_to_feature_collection(geoms: list[Any], prop_prefix: str) -> dict[str, Any]:
    """Serialize Shapely geometries to a GeoJSON FeatureCollection."""
    feats: list[dict[str, Any]] = []
    for i, g in enumerate(geoms):
        try:
            feats.append(
                {
                    "type": "Feature",
                    "geometry": shp_mapping(g),
                    "properties": {f"{prop_prefix}_ix": i},
                }
            )
        except Exception:
            continue
    return _json_safe({"type": "FeatureCollection", "features": feats})


def _footprint_feature(chip_id: str, row: pd.Series, geom_col: str) -> dict[str, Any]:
    raw = row[geom_col]
    chip = raw if hasattr(raw, "geom_type") else shape(raw)
    return _json_safe(
        {
            "type": "Feature",
            "geometry": shp_mapping(chip),
            "properties": {"chip_id": chip_id},
        }
    )


def _find_split_column(df: pd.DataFrame) -> str | None:
    for name in ("split", "Split", "split_name"):
        if name in df.columns:
            return name
    return None


def _find_geometry_column(df: pd.DataFrame) -> str | None:
    for name in ("geometry", "geom"):
        if name in df.columns:
            return name
    return None


def _iter_stac_urls_from_config(obj: Any) -> list[str]:
    """Recursively collect HTTP(S) strings that look like STAC item URLs."""
    found: list[str] = []
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("http") and (
            "stac" in s or "sentinel" in s.lower() or "/items/" in s
        ):
            found.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_iter_stac_urls_from_config(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_iter_stac_urls_from_config(v))
    return found


def load_data_config_stac_urls(
    data_root: Path, country_id: str
) -> tuple[str | None, str | None]:
    """Read data_config_{country}.json and try to recover two STAC URLs (window A/B)."""
    candidates = [
        data_root / country_id / f"data_config_{country_id}.json",
        data_root / f"data_config_{country_id}.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if not path:
        return None, None
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    urls = _iter_stac_urls_from_config(cfg)
    urls = list(dict.fromkeys(urls))
    if len(urls) >= 2:
        return urls[0], urls[1]
    if len(urls) == 1:
        return urls[0], None
    return None, None


def chip_bbox_from_row(row: pd.Series, geom_col: str) -> list[float]:
    """BBox [minx, miny, maxx, maxy] in EPSG:4326 from geometry column."""
    g = row[geom_col]
    if hasattr(g, "bounds"):
        bounds = g.bounds
    else:
        bounds = shape(g).bounds
    return [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]


def _expand_bbox_min_side_meters(
    bbox: list[float], min_side_m: float = 1600.0
) -> list[float]:
    """Grow *bbox* (WGS84) so width and height are at least *min_side_m*.

    Sentinel-2 L2A at ~10 m GSD must yield a stacked image whose shorter side is
    ≥128 px for ``ftw inference run`` (see ``ftw_tools`` ``setup_inference``).
    Field chips from GeoParquet are often only a few hundred metres — without
    expansion, download clips to a raster that fails with "Input image is too small".
    """
    minx, miny, maxx, maxy = bbox
    lat_mid = (miny + maxy) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_mid))
    w_m = (maxx - minx) * m_per_deg_lon
    h_m = (maxy - miny) * m_per_deg_lat
    pad_w = max(0.0, (min_side_m - w_m) / 2.0 / m_per_deg_lon)
    pad_h = max(0.0, (min_side_m - h_m) / 2.0 / m_per_deg_lat)
    if pad_w == 0.0 and pad_h == 0.0:
        return bbox
    return [minx - pad_w, miny - pad_h, maxx + pad_w, maxy + pad_h]


def _clip_geoms_to_chip_geometry(
    geoms: list[Any], row: pd.Series, geom_col: str
) -> list[Any]:
    """Intersect prediction geometries with the chip polygon (eval is chip-local)."""
    if geom_col not in row.index or row[geom_col] is None:
        return geoms
    raw = row[geom_col]
    chip = raw if hasattr(raw, "intersects") else shape(raw)
    out: list[Any] = []
    for g in geoms:
        try:
            if not chip.intersects(g):
                continue
            inter = g.intersection(chip)
        except Exception:
            continue
        if inter.is_empty:
            continue
        if inter.geom_type == "Polygon":
            out.append(inter)
        elif inter.geom_type == "MultiPolygon":
            out.extend(inter.geoms)
        elif inter.geom_type == "GeometryCollection":
            for part in inter.geoms:
                if part.geom_type == "Polygon":
                    out.append(part)
    return out


def _instance_mask_to_polygons(mask_path: Path) -> list[Any]:
    """Vectorize instance labels from a GeoTIFF (integer instance IDs)."""
    import numpy as np
    import rasterio
    from rasterio import features as rio_features
    from shapely.geometry import shape as shp_shape

    polys: list[Any] = []
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        transform = src.transform
        for v in np.unique(arr):
            if v <= 0:
                continue
            mask = arr == v
            for geom, val in rio_features.shapes(
                mask.astype(np.uint8), mask=mask, transform=transform
            ):
                if val:
                    polys.append(shp_shape(geom))
    return polys


def _resolve_chip_filenames(row: pd.Series, country_id: str) -> str:
    """Best-effort chip id for label mask filename."""
    for key in ("chip_id", "aoi_id", "id", "grid_id", "cell_id"):
        if key in row.index and pd.notna(row[key]):
            return str(row[key])
    return f"{country_id}_chip"


def _row_stac_urls(row: pd.Series) -> tuple[str | None, str | None]:
    """Optional per-chip STAC URLs from parquet columns."""
    for a_key, b_key in (
        ("window_a", "window_b"),
        ("win_a", "win_b"),
        ("stac_window_a", "stac_window_b"),
    ):
        if a_key in row.index and pd.notna(row.get(a_key)):
            a = str(row[a_key])
            b = str(row[b_key]) if b_key in row.index and pd.notna(row.get(b_key)) else None
            return a, b
    return None, None


def _demo_result(model_ids: list[str], country_ids: list[str], split: str) -> dict[str, Any]:
    """Synthetic scores for CI / demos when no data root is mounted."""
    return {
        "demo": True,
        "split": split,
        "message": "Synthetic benchmark result (mount FTW data under benchmark.data_root for real evaluation).",
        "by_model": {
            mid: {
                "countries": {
                    cid: {
                        "title": cid.title(),
                        "chips_evaluated": 0,
                        "scores": {
                            "finding_fields": 82.5,
                            "not_finding_non_fields": 88.0,
                            "correct_sizes_and_shapes": 76.25,
                        },
                    }
                    for cid in country_ids
                }
            }
            for mid in model_ids
        },
    }


async def run_benchmark_job(
    inference_service: InferenceService,
    *,
    model_ids: list[str],
    country_ids: list[str],
    split: str = "test",
    max_chips: int = 10,
    seed: int | None = None,
    iou_threshold: float = 0.25,
    data_root: Path | None = None,
    allow_demo: bool = False,
    auto_download: bool = False,
    include_map_geojson: bool | None = True,
) -> dict[str, Any]:
    """Run evaluation across selected countries and models.

    When *auto_download* is True, missing country data (chips parquet and the
    label-mask GeoTIFFs for the sampled chips) is fetched from Source Cooperative
    before inference begins.  *data_root* (or the built-in default cache) is used
    as the local cache directory; it is created automatically if needed.

    Requires ``data_root`` pointing to an extracted FTW tree unless ``allow_demo``
    or ``auto_download`` is True.
    """
    if seed is not None:
        random.seed(seed)

    if include_map_geojson is None:
        include_map_geojson = True

    settings = get_settings()
    root = data_root
    if root is None:
        br = getattr(settings, "benchmark", None)
        if br:
            # Prefer explicit cache_dir, then data_root, then auto-download default.
            cache = getattr(br, "cache_dir", None) or getattr(br, "data_root", None)
            if cache:
                root = Path(str(cache)).expanduser()
            elif auto_download:
                root = _DEFAULT_CACHE

    # A stale or typo'd BENCHMARK__DATA_ROOT is often non-None but not a directory.
    # In that case we must not skip the auto-download cache (chips under PROJECT_ROOT).
    if auto_download:
        if root is None or not root.is_dir():
            root = _DEFAULT_CACHE
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Auto-download: fetch missing country assets from Source Cooperative  #
    # ------------------------------------------------------------------ #
    downloads_performed: list[str] = []

    if auto_download:
        for country_id in country_ids:
            cid = country_id.lower().strip()
            try:
                parquet_path = await ensure_chips_parquet(cid, root)
                # Also try to grab the STAC-URL config (best effort – pipeline
                # falls back to per-row columns in the parquet if absent).
                await ensure_data_config(cid, root)

                # Read the parquet now to learn which chip IDs we'll need,
                # so we download only those masks rather than all of them.
                df_pre = gpd.read_parquet(parquet_path)
                split_col_pre = _find_split_column(df_pre)
                if split_col_pre:
                    raw = split.lower()
                    aliases = {raw, "validation" if raw in {"val", "validation"} else raw}
                    if raw == "val":
                        aliases.add("validation")
                    df_pre = df_pre[
                        df_pre[split_col_pre].astype(str).str.lower().isin(
                            {a.lower() for a in aliases}
                        )
                    ]
                df_pre = (
                    df_pre.sample(frac=1.0, random_state=seed)
                    if seed is not None
                    else df_pre
                )
                df_pre = df_pre.head(max_chips)
                chip_ids_needed = [
                    _resolve_chip_filenames(row, cid)
                    for _, row in df_pre.iterrows()
                ]
                if chip_ids_needed:
                    failed = await ensure_label_masks(cid, chip_ids_needed, root)
                    fetched = len(chip_ids_needed) - len(failed)
                    if fetched:
                        downloads_performed.append(
                            f"{cid}: {fetched} mask(s) downloaded"
                        )
            except ChipsParquetNotFoundError:
                raise
            except Exception as exc:
                logger.warning(
                    "Auto-download step failed for country",
                    extra={"country": country_id, "error": str(exc)},
                )

    # ------------------------------------------------------------------ #
    # Fallback: demo mode or hard error when no data is available          #
    # ------------------------------------------------------------------ #
    if root is None or not root.is_dir():
        if allow_demo:
            return _demo_result(model_ids, country_ids, split)
        hint = (
            " If you set BENCHMARK__DATA_ROOT in the system environment, it must "
            "point at an existing folder (or unset it and use auto-download)."
        )
        raise ValueError(
            "FTW benchmark data root is not configured or does not exist "
            f"(resolved root={root!s}, auto_download={auto_download}). "
            "Set BENCHMARK__DATA_ROOT to a real directory, "
            "BENCHMARK__AUTO_DOWNLOAD=true to use the project cache, "
            "or BENCHMARK__ALLOW_DEMO=true for synthetic scores."
            + (hint if root is not None else "")
        )

    polygon_defaults = {"simplify": 15, "min_size": 500, "close_interiors": False}

    default_win_a: str | None = None
    default_win_b: str | None = None

    out: dict[str, Any] = {
        "split": split,
        "max_chips": max_chips,
        "iou_threshold": iou_threshold,
        "by_model": {},
    }
    if downloads_performed:
        out["auto_downloaded"] = downloads_performed

    include_map_effective = bool(include_map_geojson) and (
        max_chips <= _MAP_GEOJSON_MAX_CHIPS
    )
    if include_map_geojson and max_chips > _MAP_GEOJSON_MAX_CHIPS:
        out["map_geojson_omitted"] = (
            f"max_chips ({max_chips}) exceeds {_MAP_GEOJSON_MAX_CHIPS}; "
            "lower max_chips or set include_map_geojson to false."
        )

    for model_id in model_ids:
        out["by_model"][model_id] = {"countries": {}}

    for country_id in country_ids:
        cid = country_id.lower().strip()
        meta = get_country_by_id(cid)
        if not meta:
            raise ValueError(f"Unknown benchmark country: {country_id}")

        chips_path = root / cid / f"chips_{cid}.parquet"
        if not chips_path.is_file():
            raise FileNotFoundError(
                f"Missing chips parquet for {cid}: {chips_path}. "
                "Ensure FTW country folder is extracted under the data root."
            )

        df = gpd.read_parquet(chips_path)
        split_col = _find_split_column(df)
        geom_col = _find_geometry_column(df)
        if not geom_col:
            raise ValueError(f"No geometry column in {chips_path}")

        if split_col:
            raw = split.lower()
            aliases = {raw}
            if raw == "validation":
                aliases.update({"val", "validation"})
            elif raw == "val":
                aliases.update({"val", "validation"})
            df = df[
                df[split_col].astype(str).str.lower().isin({a.lower() for a in aliases})
            ]
        df = df.sample(frac=1.0, random_state=seed) if seed is not None else df
        df = df.head(max_chips)
        if len(df) == 0:
            raise ValueError(
                f"No chips left after split={split!r} for {cid}. "
                "Check parquet split labels or choose another split."
            )

        default_win_a, default_win_b = load_data_config_stac_urls(root, cid)
        benchmark_year = int(meta.get("year", 2021))

        for model_id in model_ids:
            country_chips: list[ChipMetrics] = []
            map_chips: list[dict[str, Any]] = []

            for _, row in df.iterrows():
                bbox_chip = chip_bbox_from_row(row, geom_col)
                # Wider AOI for STAC download/inference — tiny field chips are <128 px at 10 m GSD.
                bbox_inf = _expand_bbox_min_side_meters(bbox_chip)
                row_a, row_b = _row_stac_urls(row)
                win_a = row_a or default_win_a
                win_b = row_b if row_a else default_win_b
                if not win_a:
                    ss = await inference_service.try_scene_selection_stac(
                        bbox_inf, benchmark_year
                    )
                    if ss:
                        win_a = ss["window_a"]
                        win_b = ss.get("window_b") or win_b
                if not win_a:
                    spec = MODEL_REGISTRY.get(model_id)
                    needs_two = bool(spec and getattr(spec, "requires_window", False))
                    wa, wb = await asyncio.to_thread(
                        pc_sentinel2_item_urls_for_bbox,
                        bbox_inf,
                        benchmark_year,
                        need_two_temporal_windows=needs_two,
                    )
                    if wa:
                        win_a = wa
                        win_b = wb or win_b
                if not win_a:
                    logger.warning(
                        "No STAC URL for chip; skipping",
                        extra={"country": cid, "row": str(row.get("id", ""))},
                    )
                    continue

                inf_params: dict[str, Any] = {
                    "model": model_id,
                    "images": [win_a] if not win_b else [win_a, win_b],
                    "bbox": bbox_inf,
                    # resize_factor 2 shrinks small chips below ftw's 128 px minimum; 1 is safer here.
                    "resize_factor": 1,
                }
                try:
                    raw = await inference_service.run_example(
                        inf_params,
                        polygon_defaults,
                        ndjson=False,
                        gpu=settings.processing.gpu,
                    )
                except Exception as e:
                    logger.warning(
                        "Inference failed for chip",
                        extra={"country": cid, "model": model_id, "error": str(e)},
                    )
                    continue

                pred_features = geojson_dict_to_features(
                    raw if isinstance(raw, dict) else {}
                )
                pred_geoms = features_to_geometries(pred_features)
                pred_geoms = _clip_geoms_to_chip_geometry(
                    pred_geoms, row, geom_col
                )

                base = _resolve_chip_filenames(row, cid)
                mask_candidates = [
                    root / cid / "label_masks" / "instance" / f"{base}.tif",
                    root / cid / "label_masks" / "instance" / f"{base.split('_')[-1]}.tif",
                ]
                gt_path = next((p for p in mask_candidates if p.is_file()), None)
                if not gt_path:
                    logger.warning(
                        "GT mask not found, skipping chip",
                        extra={"candidates": [str(p) for p in mask_candidates]},
                    )
                    continue

                gt_geoms = _instance_mask_to_polygons(gt_path)
                if not gt_geoms and not pred_geoms:
                    continue
                if include_map_effective:
                    try:
                        map_chips.append(
                            _json_safe(
                                {
                                    "chip_id": base,
                                    "footprint": _footprint_feature(
                                        base, row, geom_col
                                    ),
                                    "predictions": _geoms_to_feature_collection(
                                        pred_geoms, "pred"
                                    ),
                                    "ground_truth": _geoms_to_feature_collection(
                                        gt_geoms, "gt"
                                    ),
                                }
                            )
                        )
                    except Exception:
                        logger.exception(
                            "benchmark map geojson failed for chip",
                            extra={"country": cid, "chip": base},
                        )
                cm = greedy_match_metrics(
                    pred_geoms, gt_geoms, iou_threshold=iou_threshold
                )
                country_chips.append(cm)

            scores = aggregate_to_scores_100(country_chips)
            country_out: dict[str, Any] = {
                "title": meta["title"],
                "chips_evaluated": len(country_chips),
                "scores": {
                    "finding_fields": round(scores.finding_fields, 2),
                    "not_finding_non_fields": round(scores.not_finding_non_fields, 2),
                    "correct_sizes_and_shapes": round(scores.correct_sizes_and_shapes, 2),
                },
            }
            if len(country_chips) == 0 and len(df) > 0:
                country_out["note"] = (
                    "No chips were scored after sampling. Typical causes: STAC blocked (HTTP 403), "
                    "two-window Sentinel scenes on different tiles, inference/download errors "
                    "(see logs — e.g. 'Input image is too small' before bbox expansion), or a "
                    "missing mask. Mounting the full FTW release under BENCHMARK__DATA_ROOT "
                    "uses release STAC URLs and chip-aligned imagery."
                )
            if (
                include_map_effective
                and len(country_chips) > 0
                and len(map_chips) == 0
            ):
                country_out["map_geojson_note"] = (
                    "Scores were computed but no map layers were serialized (check API logs). "
                    "Upgrade the API if map_geojson is missing from the top-level result."
                )
            out["by_model"][model_id]["countries"][cid] = country_out

            if include_map_effective and map_chips:
                out.setdefault("map_geojson", {})
                out["map_geojson"].setdefault(model_id, {})[cid] = {"chips": map_chips}

    return out
