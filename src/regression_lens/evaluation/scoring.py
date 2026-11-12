"""Score prediction files against the generated dataset's ground truth.

Every model, whatever its architecture, writes one JSON object per pair in the
format described in docs/prediction-format.md. This module checks those
predictions, joins them with the metadata records by ``sample_id``, and
computes the shared metrics, so all models are scored by the same code.

Ground truth always comes from the metadata, never from the prediction file.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from regression_lens.evaluation.metrics import best_f1_threshold, box_iou, point_in_box, summarize

FORMAT_VERSION = "1"
MAX_REPORTED_PROBLEMS = 20


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines prediction file, skipping blank lines."""

    predictions = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error.msg})") from error
    return predictions


def write_predictions(path: str | Path, predictions: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")


def _finite_numbers(value: object, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
    )


def validate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
    operators: Sequence[str],
    expected_ids: Iterable[str] | None = None,
) -> None:
    """Raise ``ValueError`` listing every format or coverage problem found."""

    problems = []
    seen = set()
    for index, p in enumerate(predictions):
        where = f"prediction {index} ({p.get('sample_id', '?')})"
        sample_id = p.get("sample_id")
        if not isinstance(sample_id, str):
            problems.append(f"{where}: sample_id must be a string")
            continue
        if sample_id in seen:
            problems.append(f"{where}: duplicate sample_id")
        seen.add(sample_id)
        if sample_id not in records_by_id:
            problems.append(f"{where}: sample_id not in the dataset metadata")
        score = p.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            problems.append(f"{where}: score must be a finite number")
        if p.get("op_pred") is not None and p["op_pred"] not in operators:
            problems.append(f"{where}: op_pred {p['op_pred']!r} is not a known operator")
        if p.get("box") is not None and not (_finite_numbers(p["box"], 4) and p["box"][2] > 0 and p["box"][3] > 0):
            problems.append(f"{where}: box must be [x, y, w, h] with positive w and h")
        if p.get("point") is not None and not _finite_numbers(p["point"], 2):
            problems.append(f"{where}: point must be [x, y]")

    if expected_ids is not None:
        expected = set(expected_ids)
        missing, extra = expected - seen, seen - expected
        if missing:
            problems.append(f"{len(missing)} expected pairs have no prediction, e.g. {sorted(missing)[:3]}")
        if extra:
            problems.append(f"{len(extra)} predictions are outside the evaluated subset, e.g. {sorted(extra)[:3]}")

    if problems:
        shown = problems[:MAX_REPORTED_PROBLEMS]
        more = len(problems) - len(shown)
        raise ValueError("\n".join(shown + ([f"... and {more} more"] if more else [])))


def join_with_ground_truth(
    predictions: Iterable[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
    operators: Sequence[str],
) -> list[dict[str, Any]]:
    """Attach labels and localization outcomes to validated predictions."""

    joined = []
    for p in predictions:
        r = records_by_id[p["sample_id"]]
        row: dict[str, Any] = {"sample_id": p["sample_id"], "label": r["label"], "score": float(p["score"])}
        if r["label"] == "regression":
            box, point = p.get("box"), p.get("point")
            if point is None and box is not None:
                point = [box[0] + box[2] / 2, box[1] + box[3] / 2]
            # Reflowed mutations move everything below them, so point at the
            # mutated element itself rather than the whole moved region.
            target = r["bbox_after"] if r["reflow"] else r["diff_bbox"]
            row.update(
                op_true=operators.index(r["operator"]),
                op_pred=operators.index(p["op_pred"]) if p.get("op_pred") is not None else None,
                iou=box_iou(box, r["diff_bbox"]) if box is not None else None,
                pointing_hit=point_in_box(point, target) if point is not None else None,
                family=r["family"],
                magnitude=r["magnitude"],
                reflow=r["reflow"],
            )
        joined.append(row)
    return joined


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]],
    operators: Sequence[str],
    threshold: float | None = None,
    threshold_selection: str | None = None,
    expected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate, join and summarize predictions.

    With no ``threshold``, the threshold that maximizes F1 on these same
    predictions is used. That is right for validation, but optimistic
    anywhere else, so pass a threshold chosen on validation when scoring test
    or external data.
    """

    records_by_id = {r["sample_id"]: r for r in records}
    validate_predictions(predictions, records_by_id, operators, expected_ids)
    joined = join_with_ground_truth(predictions, records_by_id, operators)
    if threshold is None:
        threshold = best_f1_threshold(
            np.array([row["label"] == "regression" for row in joined]),
            np.array([row["score"] for row in joined]),
        )
        threshold_selection = "max F1 on the scored predictions"
    return {
        "format_version": FORMAT_VERSION,
        "threshold_selection": threshold_selection or "given",
        **summarize(joined, threshold, operators),
    }
