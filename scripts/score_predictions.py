"""Score one or more prediction files with the team's shared metrics.

Predictions must follow docs/prediction-format.md. Ground truth comes from the
dataset metadata, and every file must cover exactly the evaluated subset.

Examples:
  # Score one model's validation predictions (threshold = max F1 on them)
  python scripts/score_predictions.py outputs/uiclip_frozen/best_auroc/val_predictions.jsonl

  # Every run's best-AUROC checkpoint
  python scripts/score_predictions.py outputs/*/best_auroc/val_predictions.jsonl

  # Compare several models side by side and save the results
  python scripts/score_predictions.py preds/uiclip.jsonl preds/dinov3.jsonl preds/bit.jsonl \
      --out outputs/comparison

  # Score test predictions with thresholds chosen on each model's validation
  # predictions (one --threshold-from file per predictions file, same order)
  python scripts/score_predictions.py preds/uiclip_test.jsonl preds/dinov3_test.jsonl --subset test \
      --threshold-from preds/uiclip_val.jsonl preds/dinov3_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from regression_lens.data.pairs import load_records, operator_names
from regression_lens.evaluation.metrics import best_f1_threshold
from regression_lens.evaluation.scoring import load_predictions, score_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", help="JSON Lines prediction files")
    parser.add_argument("--names", nargs="+", help="display names (default: parent folder or file stem)")
    parser.add_argument("--data-root", default="d2c_v2")
    parser.add_argument("--split", default="data/processed/splits_placeholder_85_15.json")
    parser.add_argument("--subset", default="val",
                        help="split key the predictions must cover, or 'all' for every pair")
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--threshold", nargs="+", type=float,
                            help="fixed decision threshold(s), one per file or one for all")
    thresholds.add_argument("--threshold-from", nargs="+",
                            help="validation prediction file(s); use each one's max-F1 threshold")
    parser.add_argument("--out", help="directory for <name>.metrics.json (/ in names becomes __) and comparison.md")
    return parser.parse_args()


def display_name(path: Path) -> str:
    """``outputs/uiclip_frozen/best_auroc/val_predictions.jsonl`` -> ``uiclip_frozen/best_auroc``.

    Prediction files named ``*predictions`` take their folder's name; a
    ``best_*`` folder is prefixed with its run's name.
    """

    if not path.stem.endswith("predictions") or not path.parent.name:
        return path.stem
    if path.parent.name.startswith("best_") and path.parent.parent.name:
        return f"{path.parent.parent.name}/{path.parent.name}"
    return path.parent.name


def per_file(values: list[Any] | None, n: int, flag: str) -> list[Any]:
    if values is None:
        return [None] * n
    if len(values) == 1:
        return values * n
    if len(values) != n:
        sys.exit(f"{flag} needs one value, or one per predictions file ({n})")
    return values


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


COLUMNS = [
    ("AUROC reg/benign", lambda m: m["auroc"]["regression_vs_benign"]),
    ("benign FPR@95R", lambda m: (m["benign_fpr_at_95_recall"] or {}).get("fpr")),
    ("threshold", lambda m: m["threshold"]),
    ("precision", lambda m: m["detection"]["precision"]),
    ("recall", lambda m: m["detection"]["recall"]),
    ("F1", lambda m: m["detection"]["f1"]),
    ("benign FPR", lambda m: m["false_positive_rate"].get("benign")),
    ("no-change FPR", lambda m: m["false_positive_rate"].get("no_change")),
    ("type macro-F1", lambda m: m.get("type", {}).get("macro_f1")),
    ("pointing", lambda m: m.get("localization", {}).get("pointing_accuracy")),
    ("mean IoU", lambda m: m.get("localization", {}).get("mean_iou")),
]


def comparison_table(results: dict[str, dict[str, Any]]) -> str:
    header = ["model", "n"] + [name for name, _ in COLUMNS]
    rows = [[name, str(m["n"])] + [fmt(get(m)) for _, get in COLUMNS] for name, m in results.items()]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.predictions]
    names = args.names or [display_name(p) for p in paths]
    if len(names) != len(paths) or len(set(names)) != len(names):
        sys.exit("--names must give one unique name per predictions file")

    records = load_records(args.data_root)
    all_records_by_id = {r["sample_id"]: r for r in records}
    operators = operator_names(records)
    expected_ids = None
    if args.subset != "all":
        split_path = Path(args.split)
        if not split_path.exists():
            sys.exit(f"split file {split_path} not found; pass --split or use --subset all")
        split = json.loads(split_path.read_text())
        if args.subset not in split:
            sys.exit(f"split file has no {args.subset!r} key (has: {', '.join(split)})")
        pages = set(split[args.subset])
        expected_ids = [r["sample_id"] for r in records if r["parent_id"] in pages]
        records = [r for r in records if r["parent_id"] in pages]

    fixed = per_file(args.threshold, len(paths), "--threshold")
    sources = per_file(args.threshold_from, len(paths), "--threshold-from")

    results = {}
    for name, path, threshold, source in zip(names, paths, fixed, sources):
        selection = f"fixed {threshold}" if threshold is not None else None
        if source is not None:
            val = [p for p in load_predictions(source) if p.get("sample_id") in all_records_by_id]
            if not val:
                sys.exit(f"{source}: no predictions match the dataset metadata")
            threshold = best_f1_threshold(
                np.array([all_records_by_id[p["sample_id"]]["label"] == "regression" for p in val]),
                np.array([float(p["score"]) for p in val]),
            )
            selection = f"max F1 on {source}"
        try:
            results[name] = {
                "predictions": str(path),
                **score_predictions(load_predictions(path), records, operators, threshold, selection, expected_ids),
            }
        except ValueError as error:
            sys.exit(f"{path}: invalid predictions\n{error}")

    table = comparison_table(results)
    print(table)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, metrics in results.items():
            # Names like "run/best_auroc" become "run__best_auroc.metrics.json".
            (out / f"{name.replace('/', '__')}.metrics.json").write_text(json.dumps(metrics, indent=2))
        (out / "comparison.md").write_text(
            f"Subset: {args.subset} ({args.split})\n\n{table}\n\n"
            + "\n".join(f"- {name}: threshold {m['threshold_selection']}" for name, m in results.items())
            + "\n"
        )


if __name__ == "__main__":
    main()
