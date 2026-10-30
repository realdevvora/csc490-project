"""Page-level train/validation splits for the generated Design2Code dataset.

Every variant of a page (mutations, re-render, browser renders) shares the
page's ``parent_id``, so splitting on ``parent_id`` keeps all of them in the
same partition (see ADR 0003).

The default split produced here is a placeholder: a seeded 85/15 shuffle. Pass
a real split file to ``load_or_create_split`` once the team has frozen one.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PLACEHOLDER_SEED = 490
PLACEHOLDER_VAL_FRACTION = 0.15


def page_ids(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return the sorted, unique source-page ids in ``records``."""

    return sorted({record["parent_id"] for record in records})


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


def load_or_create_split(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Load a ``{"train": [...], "val": [...]}`` page split, creating it if absent.

    A newly created split is written to ``path`` so later runs reuse it.
    """

    path = Path(path)
    records = list(records)
    if path.exists():
        split = json.loads(path.read_text())
    else:
        split = make_page_split(records)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split, indent=2))

    train, val = set(split["train"]), set(split["val"])
    if train & val:
        raise ValueError(f"{path}: pages appear in both train and val")
    missing = set(page_ids(records)) - train - val
    if missing:
        raise ValueError(f"{path}: {len(missing)} pages are in neither split")
    return {"train": sorted(train), "val": sorted(val)}
