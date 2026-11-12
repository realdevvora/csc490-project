import json
from pathlib import Path

import pytest

from regression_lens.data.splits import (
    TEAM_SPLIT,
    broken_no_change_pages,
    load_or_create_split,
    load_split,
    make_page_split,
    make_team_split,
)

DATASET = Path(__file__).parents[1] / "d2c_v2" / "master_dataset_with_benign.json"


def _records(n_pages: int, broken: frozenset[int] = frozenset()) -> list[dict]:
    """Three pairs per page; the ``no_change`` pair of each ``broken`` page differs."""

    records = []
    for page in range(n_pages):
        for variant, label in (("identical", "no_change"), ("firefox", "benign"), ("font_size", "regression")):
            changed = label != "no_change" or page in broken
            records.append({
                "parent_id": f"d2c_{page:03d}",
                "sample_id": f"d2c_{page:03d}_{variant}",
                "label": label,
                "size_ref": [1280, 800],
                "size_cand": [1280, 800],
                "changed_px": 100 if changed else 0,
            })
    return records


def test_placeholder_split_is_page_disjoint_and_deterministic() -> None:
    records = _records(100)
    split = make_page_split(records)

    assert len(split["val"]) == 15
    assert len(split["train"]) == 85
    assert not set(split["train"]) & set(split["val"])
    assert split == make_page_split(records)


def test_load_or_create_split_writes_then_reuses(tmp_path) -> None:
    path = tmp_path / "split.json"
    records = _records(20)

    created = load_or_create_split(path, records)
    assert path.exists()
    assert load_or_create_split(path, records) == created


def test_load_or_create_split_rejects_overlap(tmp_path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["d2c_000", "d2c_001"], "val": ["d2c_001"]}))

    with pytest.raises(ValueError, match="both"):
        load_or_create_split(path, _records(2))


def test_load_split_requires_the_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path / "missing.json", _records(2))


def test_broken_no_change_pages_checks_size_and_pixels() -> None:
    records = _records(4, broken=frozenset({1}))
    records[6]["size_cand"] = [1280, 806]  # d2c_002_identical: taller, but no changed pixels

    assert broken_no_change_pages(records) == ["d2c_001", "d2c_002"]


def test_team_split_excludes_broken_pages_and_keeps_proportions() -> None:
    records = _records(100, broken=frozenset(range(0, 100, 5)))
    split = make_team_split(records)

    assert split["excluded"] == [f"d2c_{page:03d}" for page in range(0, 100, 5)]
    assert len(split["val"]) == 12 and len(split["train"]) == 68
    assert not set(split["excluded"]) & (set(split["train"]) | set(split["val"]))
    assert split == make_team_split(records)


def test_load_split_accepts_excluded_pages(tmp_path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["d2c_000"], "val": ["d2c_001"], "excluded": ["d2c_002"]}))

    assert load_split(path, _records(3)) == {"train": ["d2c_000"], "val": ["d2c_001"]}


def test_load_split_rejects_excluded_page_in_a_partition(tmp_path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["d2c_000"], "val": ["d2c_001"], "excluded": ["d2c_001"]}))

    with pytest.raises(ValueError, match="excluded"):
        load_split(path, _records(2))


def test_team_split_is_page_disjoint_and_complete() -> None:
    split = json.loads((Path(__file__).parents[1] / TEAM_SPLIT).read_text())
    train, val, excluded = set(split["train"]), set(split["val"]), set(split["excluded"])

    assert len(train) == 390 and len(val) == 69 and len(excluded) == 25
    assert not train & val and not excluded & (train | val)
    assert len(train | val | excluded) == 484


@pytest.mark.skipif(not DATASET.exists(), reason="local dataset d2c_v2/ not available")
def test_team_split_file_matches_the_dataset() -> None:
    records = json.loads(DATASET.read_text())
    committed = json.loads((Path(__file__).parents[1] / TEAM_SPLIT).read_text())

    assert committed == make_team_split(records)
