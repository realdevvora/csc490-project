import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from regression_lens.evaluation.metrics import auroc, fpr_at_recall  # noqa: E402
from regression_lens.evaluation.scoring import score_predictions  # noqa: E402

OPERATORS = ["font_size", "margin"]


def _records() -> list[dict]:
    regression = {"label": "regression", "family": "typography", "magnitude": "small"}
    return [
        {"sample_id": "a", "operator": "font_size", "diff_bbox": [0, 0, 10, 10],
         "bbox_after": [0, 0, 10, 10], "reflow": False, **regression},
        {"sample_id": "b", "operator": "margin", "diff_bbox": [0, 0, 100, 100],
         "bbox_after": [50, 50, 10, 10], "reflow": True, **regression},
        {"sample_id": "c", "label": "benign", "operator": None},
        {"sample_id": "d", "label": "no_change", "operator": None},
    ]


def test_auroc_handles_ties_and_missing_classes() -> None:
    assert auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc([1, 0], [0.5, 0.5]) == 0.5
    assert auroc([1, 1], [0.5, 0.6]) is None


def test_fpr_at_recall_uses_recall_threshold() -> None:
    result = fpr_at_recall([0.9, 0.8, 0.7, 0.6], [0.65, 0.1], recall=0.75)

    assert result == {"threshold": 0.7, "recall": 0.75, "fpr": 0.0}


def test_score_predictions_joins_ground_truth() -> None:
    predictions = [
        {"sample_id": "a", "score": 0.9, "op_pred": "font_size", "box": [0, 0, 10, 10]},
        {"sample_id": "b", "score": 0.8, "op_pred": "font_size", "point": [55, 55]},
        {"sample_id": "c", "score": 0.85},
        {"sample_id": "d", "score": 0.1},
    ]

    metrics = score_predictions(predictions, _records(), OPERATORS, threshold=0.5)

    assert metrics["auroc"]["regression_vs_benign"] == 0.5
    assert metrics["false_positive_rate"] == {"benign": 1.0, "no_change": 0.0}
    assert metrics["type"]["n"] == 2
    # "a": box centre (5, 5) is inside diff_bbox; "b": reflowed, point inside bbox_after.
    assert metrics["localization"] == {"n_box": 1, "mean_iou": 1.0, "n_point": 2, "pointing_accuracy": 1.0}


def test_score_predictions_reports_bad_and_missing_predictions() -> None:
    predictions = [
        {"sample_id": "a", "score": "high"},
        {"sample_id": "a", "score": 0.5, "op_pred": "kerning"},
        {"sample_id": "zzz", "score": 0.5, "box": [0, 0, 0, 5]},
    ]

    with pytest.raises(ValueError) as error:
        score_predictions(predictions, _records(), OPERATORS, expected_ids=["a", "b", "c", "d"])

    message = str(error.value)
    for problem in ("finite number", "duplicate", "not a known operator", "not in the dataset",
                    "positive w and h", "have no prediction"):
        assert problem in message
