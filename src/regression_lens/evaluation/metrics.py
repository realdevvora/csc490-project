"""Metrics from the experiment protocol (docs/experiment-protocol.md).

Detection:     precision, recall, F1, false-positive rate on benign/no-change,
               AUROC, and benign false-positive rate at 95% recall
Localization:  bounding-box IoU and pointing accuracy
Category:      macro F1 over mutation operators, with per-class F1 and support

Boxes are ``[x, y, w, h]`` in original screenshot pixels, matching the
dataset's ``diff_bbox`` and ``bbox_after`` fields.

Requires numpy and scipy (the ``uiclip`` extra).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

Box = Sequence[float]


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Binary precision, recall and F1 with regression as the positive class."""

    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold on ``scores`` that maximizes F1. Select it on validation data only."""

    candidates = np.unique(scores)
    f1s = [detection_metrics(y_true, scores >= t)["f1"] for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Area under the ROC curve: P(random positive outscores random negative).

    Ties count as half. Returns ``None`` when either class is absent.
    """

    y_true = np.asarray(y_true, bool)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(scores)
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fpr_at_recall(
    positive_scores: np.ndarray, negative_scores: np.ndarray, recall: float = 0.95
) -> dict[str, float] | None:
    """False-positive rate at the highest threshold that reaches ``recall``.

    The threshold comes from the recall target, not from tuning, so it is
    comparable across models. Returns ``None`` when either set is empty.
    """

    positive_scores = np.sort(np.asarray(positive_scores, float))[::-1]
    negative_scores = np.asarray(negative_scores, float)
    if not len(positive_scores) or not len(negative_scores):
        return None
    threshold = positive_scores[int(np.ceil(recall * len(positive_scores))) - 1]
    return {
        "threshold": float(threshold),
        "recall": float((positive_scores >= threshold).mean()),
        "fpr": float((negative_scores >= threshold).mean()),
    }


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], names: Sequence[str]) -> dict[str, Any]:
    """Macro F1 over the classes present in ``y_true``, plus per-class F1/support."""

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = {}
    for index in np.unique(y_true):
        true, pred = y_true == index, y_pred == index
        tp = int((true & pred).sum())
        denominator = int(true.sum() + pred.sum())
        per_class[names[index]] = {
            "f1": 2 * tp / denominator if denominator else 0.0,
            "support": int(true.sum()),
        }
    scores = [c["f1"] for c in per_class.values()]
    return {"macro_f1": float(np.mean(scores)) if scores else 0.0, "per_class": per_class}


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two ``[x, y, w, h]`` boxes."""

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def point_in_box(point: Sequence[float], box: Box) -> bool:
    x, y = point
    bx, by, bw, bh = box
    return bool(bx <= x <= bx + bw and by <= y <= by + bh)


def peak_point(change_map: np.ndarray, cell_px: float) -> tuple[float, float]:
    """Centre, in original pixels, of the highest-scoring change-map cell."""

    row, col = np.unravel_index(int(np.argmax(change_map)), change_map.shape)
    return ((col + 0.5) * cell_px, (row + 0.5) * cell_px)


def largest_component_box(
    change_map: np.ndarray, threshold: float, cell_px: float
) -> list[float] | None:
    """Box around the largest connected region of ``change_map >= threshold``.

    Falls back to the single peak cell when nothing clears the threshold.
    """

    components, count = ndimage.label(change_map >= threshold)
    if count == 0:
        row, col = np.unravel_index(int(np.argmax(change_map)), change_map.shape)
        return [col * cell_px, row * cell_px, cell_px, cell_px]
    sizes = ndimage.sum(np.ones_like(change_map), components, range(1, count + 1))
    rows, cols = ndimage.find_objects(components)[int(np.argmax(sizes))]
    return [
        cols.start * cell_px,
        rows.start * cell_px,
        (cols.stop - cols.start) * cell_px,
        (rows.stop - rows.start) * cell_px,
    ]


def summarize(
    predictions: Iterable[Mapping[str, Any]],
    threshold: float,
    operators: Sequence[str],
) -> dict[str, Any]:
    """Aggregate per-pair predictions into the protocol's metrics.

    Each prediction needs ``label`` and ``score``. Regressions also need
    ``family``, ``magnitude`` and ``reflow``, and may carry ``op_true`` with
    ``op_pred``, ``iou`` and ``pointing_hit``; a ``None`` value leaves that pair
    out of the corresponding metric, and the ``n`` fields report coverage.
    """

    predictions = list(predictions)
    labels = np.array([p["label"] for p in predictions])
    scores = np.array([p["score"] for p in predictions], dtype=float)
    flagged = scores >= threshold
    is_regression = labels == "regression"
    regressions = [p for p in predictions if p["label"] == "regression"]
    vs_benign = is_regression | (labels == "benign")

    result: dict[str, Any] = {
        "threshold": threshold,
        "n": len(predictions),
        "n_by_label": {label: int((labels == label).sum()) for label in sorted(set(labels))},
        "detection": detection_metrics(is_regression, flagged),
        "false_positive_rate": {
            label: float(flagged[labels == label].mean())
            for label in ("benign", "no_change")
            if (labels == label).any()
        },
        "auroc": {
            "regression_vs_benign": auroc(is_regression[vs_benign], scores[vs_benign]),
            "regression_vs_all_negatives": auroc(is_regression, scores),
        },
        "benign_fpr_at_95_recall": fpr_at_recall(
            scores[is_regression], scores[labels == "benign"], 0.95
        ),
    }
    if regressions:
        typed = [p for p in regressions if p.get("op_pred") is not None]
        if typed:
            result["type"] = {
                "n": len(typed),
                **macro_f1([p["op_true"] for p in typed], [p["op_pred"] for p in typed], operators),
            }
        boxed = [p["iou"] for p in regressions if p.get("iou") is not None]
        pointed = [p["pointing_hit"] for p in regressions if p.get("pointing_hit") is not None]
        if boxed or pointed:
            result["localization"] = {
                "n_box": len(boxed),
                "mean_iou": float(np.mean(boxed)) if boxed else None,
                "n_point": len(pointed),
                "pointing_accuracy": float(np.mean(pointed)) if pointed else None,
            }
        result["recall_by"] = {
            key: _recall_by(regressions, key, threshold)
            for key in ("family", "magnitude", "reflow")
        }
    return result


def _recall_by(regressions: list[Mapping[str, Any]], key: str, threshold: float) -> dict[str, float]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for p in regressions:
        groups[str(p[key])].append(p["score"] >= threshold)
    return {group: float(np.mean(hits)) for group, hits in sorted(groups.items())}
