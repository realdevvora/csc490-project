# DiffSpot

A local copy of [tencent/DiffSpot](https://huggingface.co/datasets/tencent/DiffSpot), for evaluation only. See [ADR 0001](../../docs/adr/0001-training-and-external-evaluation-datasets.md): it is never used for training.

| | |
|---|---|
| Source | `tencent/DiffSpot`, `test-00000-of-00001.parquet` |
| Revision | `c6dd79d5e1c0cbb4e7ca234c9f53c418a75e30ce` |
| SHA-256 | `bd3596a2cbedaab10dbc689e202df4469ded4af498989416f7b9e01ad496a516` |
| Downloaded | 2026-11-24 |
| Licence | MIT, see [LICENSE](LICENSE) |

## Contents

- `images/<id>_before.png`, `images/<id>_after.png`: the screenshot pair.
- `metadata.jsonl`: one line per pair with every other field from the source, plus `before_path` and `after_path`.

**4,400 pairs:**
- 3,900 with a change: 13 mutation types × 300, split evenly into `easy`, `medium` and `hard`.
- 500 with no change (`no_diff`).
- 15 page domains.

**Fields** (from the dataset card):
- `ground_truth_diff`: a description of the change in words.
- `mutation_types`, `mutations_text`, `mutation_dicts_json`: the CSS change made.
- `task_type`: `visual_diff` or `no_diff`.
- `difficulty`: `easy`, `medium`, `hard` or `no_diff`.
- `domain`.
- `pixel_diff`, `target_diff`, `outside_diff`: how much changed.
- `target_bbox_x/y/w/h`: the changed element's box, `-1` when there isn't one.

DiffSpot's 13 mutation types are not the same as our 13 operators, and it has no benign pairs.

## Rebuild

```bash
pip install pyarrow
python scripts/fetch_diffspot.py
```

The script downloads the pinned revision, checks its SHA-256, and rewrites this folder.
