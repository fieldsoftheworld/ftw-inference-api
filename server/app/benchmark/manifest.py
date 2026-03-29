"""Load FTW benchmark country manifest (aligned with Source Cooperative FTW v1)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "ftw_countries.json"


@lru_cache
def load_country_manifest() -> list[dict[str, Any]]:
    """Return the static list of FTW benchmark countries."""
    with _MANIFEST_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_country_by_id(country_id: str) -> dict[str, Any] | None:
    """Return manifest row for id or None."""
    cid = country_id.lower().strip()
    for row in load_country_manifest():
        if row["id"] == cid:
            return row
    return None
