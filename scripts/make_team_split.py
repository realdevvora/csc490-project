"""Regenerate the team split, ``data/splits/splits_85_15.json``.

Pages whose ``no_change`` pair is not pixel-identical are excluded with all
their pairs; the rest are split 85/15 by source page with seed 490.

    python scripts/make_team_split.py [--data-root d2c_v2] [--out data/splits/splits_85_15.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from regression_lens.data.records import load_records
from regression_lens.data.splits import TEAM_SPLIT, make_team_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="d2c_v2", help="folder with master_dataset_with_benign.json")
    parser.add_argument("--out", default=TEAM_SPLIT, help="split JSON to write")
    args = parser.parse_args()

    records = load_records(args.data_root)
    split = make_team_split(records)
    Path(args.out).write_text(json.dumps(split, indent=2) + "\n")
    pages = {name: set(split[name]) for name in ("train", "val", "excluded")}
    for name, ids in pages.items():
        pairs = sum(record["parent_id"] in ids for record in records)
        print(f"{name:9s} {len(ids):4d} pages  {pairs:5d} pairs")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
