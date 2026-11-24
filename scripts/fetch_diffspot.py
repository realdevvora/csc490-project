"""Download DiffSpot at a pinned revision and unpack it into ``evaluation/diffspot``.

Writes ``images/<id>_before.png``, ``images/<id>_after.png`` and one
``metadata.jsonl`` line per pair with every non-image field. Needs ``pyarrow``.

    python scripts/fetch_diffspot.py [--parquet test.parquet] [--out evaluation/diffspot]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "c6dd79d5e1c0cbb4e7ca234c9f53c418a75e30ce"
URL = f"https://huggingface.co/datasets/tencent/DiffSpot/resolve/{REVISION}/test-00000-of-00001.parquet"
SHA256 = "bd3596a2cbedaab10dbc689e202df4469ded4af498989416f7b9e01ad496a516"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", help="already-downloaded parquet; downloaded if omitted")
    parser.add_argument("--out", default="evaluation/diffspot", help="folder to write")
    args = parser.parse_args()

    # Fetch the pinned file unless one is given, and refuse anything that isn't byte-identical
    parquet = Path(args.parquet) if args.parquet else Path(tempfile.mkdtemp()) / "test.parquet"
    if not args.parquet:
        urllib.request.urlretrieve(URL, parquet)
    if sha256(parquet) != SHA256:
        raise SystemExit(f"{parquet} does not match the pinned DiffSpot revision")

    out = Path(args.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(parquet).to_pylist()
    with (out / "metadata.jsonl").open("w") as meta:
        for row in rows:
            before, after = row.pop("image_before"), row.pop("image_after")
            row["before_path"] = f"images/{row['id']}_before.png"
            row["after_path"] = f"images/{row['id']}_after.png"
            (out / row["before_path"]).write_bytes(before["bytes"])
            (out / row["after_path"]).write_bytes(after["bytes"])
            meta.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
