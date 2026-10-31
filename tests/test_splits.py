import json

import pytest

from regression_lens.data.splits import load_or_create_split, make_page_split


def _records(n_pages: int) -> list[dict[str, str]]:
    return [
        {"parent_id": f"d2c_{page:03d}", "sample_id": f"d2c_{page:03d}_{variant}"}
        for page in range(n_pages)
        for variant in ("identical", "firefox", "font_size")
    ]


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
