"""Page-level train/validation splits for the generated Design2Code dataset.

Every variant of a page (mutations, re-render, browser renders) shares the
page's ``parent_id``, so splitting on ``parent_id`` keeps all of them in the
same partition (see ADR 0003).

The team's split is ``TEAM_SPLIT``; load it with ``load_split``. It is built by
``make_team_split`` (``scripts/make_team_split.py``): pages whose ``no_change``
pair is not actually identical are excluded entirely, and the remaining pages are
split 85/15 (390 train / 69 validation pages, 25 excluded). ``make_page_split``
builds the older seeded placeholder, kept for reproducing the first experiments
and for tests.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

TEAM_SPLIT = "data/splits/splits_85_15.json"
PLACEHOLDER_SEED = 490
PLACEHOLDER_VAL_FRACTION = 0.15


def page_ids(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return the sorted, unique source-page ids in ``records``."""

    return sorted({record["parent_id"] for record in records})


def broken_no_change_pages(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return the sorted pages whose ``no_change`` pair is not pixel-identical.

    A re-render that changed size or has any changed pixel is not a true
    no-change example, and the difference is not necessarily benign either (a
    colour can differ), so the whole page is left out of the split.
    """

    return sorted(
        {
            record["parent_id"]
            for record in records
            if record["label"] == "no_change"
            and (record["size_ref"] != record["size_cand"] or record["changed_px"] != 0)
        }
    )


def make_team_split(
    records: Iterable[Mapping[str, Any]],
    val_fraction: float = PLACEHOLDER_VAL_FRACTION,
    seed: int = PLACEHOLDER_SEED,
) -> dict[str, Any]:
    """Split the pages with a valid ``no_change`` pair, excluding the rest.

    Uses the same seeded shuffle as ``make_page_split`` but takes validation
    pages from the end of it, as the original team split did.
    """

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    records = list(records)
    excluded = broken_no_change_pages(records)
    pages = sorted(set(page_ids(records)) - set(excluded))
    random.Random(seed).shuffle(pages)
    n_val = round(len(pages) * val_fraction)
    return {
        "note": (
            "team split; pages whose no_change pair is not pixel-identical (different size or "
            "changed_px > 0) are excluded with all their pairs. Regenerate with scripts/make_team_split.py"
        ),
        "seed": seed,
        "val_fraction": val_fraction,
        "excluded": excluded,
        "train": sorted(pages[: len(pages) - n_val]),
        "val": sorted(pages[len(pages) - n_val :]),
    }


def make_page_split(
    records: Iterable[Mapping[str, Any]],
    val_fraction: float = PLACEHOLDER_VAL_FRACTION,
    seed: int = PLACEHOLDER_SEED,
) -> dict[str, Any]:
    """Shuffle pages with a fixed seed and hold out ``val_fraction`` of them."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    pages = page_ids(records)
    random.Random(seed).shuffle(pages)
    n_val = round(len(pages) * val_fraction)
    return {
        "note": "placeholder split; replace with the team's frozen split",
        "seed": seed,
        "val_fraction": val_fraction,
        "train": sorted(pages[n_val:]),
        "val": sorted(pages[:n_val]),
    }


def load_split(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Load and check a ``{"train": [...], "val": [...]}`` page split.

    An optional ``"excluded"`` list names pages left out of every partition.
    Other keys (notes, seeds) are ignored. Raises if the file is missing, or if
    the partitions overlap or leave out any page in ``records``.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split file {path} not found")
    return _checked(path, json.loads(path.read_text()), list(records))


def load_or_create_split(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Like ``load_split``, but write a seeded placeholder split if ``path`` is absent."""

    path = Path(path)
    records = list(records)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_page_split(records), indent=2))
    return load_split(path, records)


def _checked(path: Path, split: Mapping[str, Any], records: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    train, val, excluded = set(split["train"]), set(split["val"]), set(split.get("excluded", []))
    if train & val:
        raise ValueError(f"{path}: pages appear in both train and val")
    if excluded & (train | val):
        raise ValueError(f"{path}: excluded pages also appear in train or val")
    missing = set(page_ids(records)) - train - val - excluded
    if missing:
        raise ValueError(f"{path}: {len(missing)} pages are in neither split")
    return {"train": sorted(train), "val": sorted(val)}
