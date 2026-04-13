"""Helpers to cache GeoParquet files from Source Cooperative.

FTW publishes ``chips_{country}.parquet`` under inconsistent bucket layouts, for
example:

- ``.../fields-of-the-world/{country}/chips_{country}.parquet``
- ``.../fields-of-the-world-{country}/chips_{country}.parquet``

We try those locations; if neither exists, we raise
:class:`ChipsParquetNotFoundError`.  Full benchmark evaluation may still need
``data_config_*.json`` and ``label_masks/instance/`` beside the parquet.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import requests

from app.benchmark.manifest import get_country_by_id
from app.core.logging import get_logger

logger = get_logger(__name__)

_SOURCE_COOP_BASE = "https://data.source.coop/kerner-lab"
_CHUNK = 1 << 16  # 64 KB


class ChipsParquetNotFoundError(ValueError):
    """No ``chips_{country}.parquet`` at any known Source Cooperative URL."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _country_bucket_bases(country_id: str) -> list[str]:
    """Return possible Source Cooperative base paths for one country (order matters)."""
    cid = country_id.lower().strip()
    return [
        f"{_SOURCE_COOP_BASE}/fields-of-the-world/{cid}",
        f"{_SOURCE_COOP_BASE}/fields-of-the-world-{cid}",
    ]


def _download_file_sync(url: str, dest: Path, label: str = "") -> bool:
    """Stream-download *url* to *dest* (atomic via .tmp).

    Returns True when a new file was fetched, False when already cached.
    Raises ``requests.HTTPError`` on non-2xx status.
    """
    if dest.exists():
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info(f"Downloading {label or url}")

    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        fh.write(chunk)
        tmp.rename(dest)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


async def _download(url: str, dest: Path, label: str = "") -> bool:
    """Async wrapper: run the blocking download in a thread-pool worker."""
    return await asyncio.to_thread(_download_file_sync, url, dest, label)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def ensure_chips_parquet(country_id: str, data_root: Path) -> Path:
    """Return path to ``chips_{country_id}.parquet``, downloading if needed.

    Tries known Source Cooperative layouts; if none serve that exact filename,
    raises :class:`ChipsParquetNotFoundError` with message
    ``no file found with name chips_<country>.parquet``.
    """
    meta = get_country_by_id(country_id)
    if not meta:
        raise ValueError(f"Unknown benchmark country: {country_id!r}")

    cid = country_id.lower().strip()
    chips_name = f"chips_{cid}.parquet"
    dest = data_root / cid / chips_name

    if dest.exists():
        return dest

    for base in _country_bucket_bases(cid):
        url = f"{base}/{chips_name}"
        try:
            await _download(
                url,
                dest,
                label=f"{cid} {chips_name} ({meta['chips']} chips)",
            )
            return dest
        except requests.HTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            if code == 404:
                continue
            raise

    raise ChipsParquetNotFoundError(f"no file found with name {chips_name}")


async def ensure_label_masks(
    country_id: str,
    chip_ids: list[str],
    data_root: Path,
    *,
    concurrency: int = 4,
) -> list[str]:
    """Download label masks for *chip_ids* that are not yet cached.

    Returns the list of chip_ids whose download failed (caller should skip them).
    Downloads are run with bounded concurrency to avoid hammering the CDN.
    """
    cid = country_id.lower().strip()
    bases = _country_bucket_bases(cid)
    mask_dir = data_root / cid / "label_masks" / "instance"
    failed: list[str] = []

    sem = asyncio.Semaphore(concurrency)

    async def _fetch_one(chip_id: str) -> None:
        dest = mask_dir / f"{chip_id}.tif"
        if dest.exists():
            return
        async with sem:
            for base in bases:
                url = f"{base}/label_masks/instance/{chip_id}.tif"
                try:
                    await _download(url, dest, label=f"{cid} mask {chip_id}")
                    return
                except requests.HTTPError as exc:
                    if getattr(exc.response, "status_code", None) == 404:
                        continue
                    logger.warning(
                        "Mask download failed – chip will be skipped",
                        extra={"country": cid, "chip_id": chip_id, "error": str(exc)},
                    )
                    failed.append(chip_id)
                    return
                except Exception as exc:
                    logger.warning(
                        "Mask download failed – chip will be skipped",
                        extra={"country": cid, "chip_id": chip_id, "error": str(exc)},
                    )
                    failed.append(chip_id)
                    return
            logger.warning(
                "Mask not found on any known bucket layout – chip will be skipped",
                extra={"country": cid, "chip_id": chip_id},
            )
            failed.append(chip_id)

    await asyncio.gather(*[_fetch_one(cid_) for cid_ in chip_ids])
    return failed


async def ensure_data_config(country_id: str, data_root: Path) -> bool:
    """Try to download ``data_config_{country_id}.json`` (STAC URL fallback).

    Returns True when the file is present (cached or freshly downloaded),
    False when it is unavailable on Source Cooperative.
    """
    cid = country_id.lower().strip()
    dest = data_root / cid / f"data_config_{cid}.json"
    if dest.exists():
        return True

    for base in _country_bucket_bases(cid):
        url = f"{base}/data_config_{cid}.json"
        try:
            await _download(url, dest, label=f"{cid} data_config")
            return True
        except requests.HTTPError as exc:
            if getattr(exc.response, "status_code", None) == 404:
                continue
            logger.debug(f"data_config HTTP error for {cid}: {exc}")
            return False
        except Exception as exc:
            logger.debug(f"data_config not available for {cid}: {exc}")
            return False
    logger.debug(f"data_config not found for {cid} on any known bucket layout")
    return False
