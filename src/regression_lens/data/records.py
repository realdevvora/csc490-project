"""Load the generated dataset's pair records (``master_dataset_with_benign.json``).

Plain JSON only, so scoring and splits don't need torch.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def load_records(data_root: str | Path) -> list[dict[str, Any]]:
    """Load every pair record from ``master_dataset_with_benign.json``."""

    return json.loads((Path(data_root) / "master_dataset_with_benign.json").read_text())


def operator_names(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return the sorted mutation operators, the class order of the type head."""

    return sorted({record["operator"] for record in records if record["operator"]})
