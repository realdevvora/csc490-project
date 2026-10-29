import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from regression_lens.evaluation.metrics import (  # noqa: E402
    best_f1_threshold,
    box_iou,
    detection_metrics,
    largest_component_box,
    macro_f1,
)


def test_detection_metrics() -> None:
    metrics = detection_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_best_f1_threshold_separates_scores() -> None:
    assert best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.9])) == 0.7


def test_box_iou() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)
    assert box_iou([0, 0, 1, 1], [5, 5, 1, 1]) == 0.0


def test_macro_f1_averages_present_classes() -> None:
    result = macro_f1([0, 0, 1], [0, 1, 1], ["a", "b", "c"])

    assert set(result["per_class"]) == {"a", "b"}
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_largest_component_box_in_original_pixels() -> None:
    change_map = np.zeros((10, 10))
    change_map[2:4, 5:8] = 0.9
    change_map[8, 0] = 0.9

    assert largest_component_box(change_map, 0.5, cell_px=30) == [150, 60, 90, 60]
