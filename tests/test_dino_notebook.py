"""Check notebook distance/coordinate math without downloading or loading DINO."""

import ast
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from regression_lens.evaluation.metrics import peak_point, point_in_box


@pytest.fixture
def notebook_math():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "dino_exploration.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "dinov2-distances")
    tree = ast.parse("".join(cell["source"]))
    functions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {
        "np": np,
        "torch": torch,
        "F": torch.nn.functional,
        "math": math,
        "peak_point": peak_point,
        "point_in_box": point_in_box,
        "GRID_SIZE": 2,
        "PATCH_SIZE": 14,
        "rectangle": [100, 200, 128, 228],
    }
    exec(compile(functions, str(notebook_path), "exec"), namespace)
    return namespace


def test_patch_distances_preserve_positions_and_raw_scale(notebook_math):
    reference = torch.tensor([[[1.0, 0.0]] * 4])
    candidate = torch.tensor([[[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])

    maps = notebook_math["distance_maps"](reference, candidate)

    np.testing.assert_allclose(maps["cosine"], [[1, 2], [0, 0]])
    np.testing.assert_allclose(maps["l2"], [[2**0.5, 2], [0, 0]], rtol=1e-6)


def test_roi_overlap_maps_original_coordinates_to_fractional_patches(notebook_math):
    overlap = notebook_math["roi_patch_overlap"]([107, 200, 14, 14])

    np.testing.assert_allclose(overlap, [[0.5, 0.5], [0, 0]])
    outside = notebook_math["roi_patch_overlap"]([0, 0, 10, 10])
    assert not outside.any()
    moved_origin = notebook_math["roi_patch_overlap"]([7, 0, 14, 14], [0, 0, 28, 28])
    np.testing.assert_allclose(moved_origin, overlap)


def test_roi_mean_respects_coverage_padding_and_empty_regions(notebook_math):
    values = np.array([[2.0, 8.0], [20.0, 80.0]])
    coverage = np.array([[0.5, 0.5], [0.0, 0.0]])
    valid = np.array([[True, False], [True, True]])

    weighted_mean = notebook_math["weighted_mean"]
    assert weighted_mean(values, coverage) == 5
    assert weighted_mean(values, coverage * valid) == 2
    assert weighted_mean(values, np.zeros_like(coverage)) is None


def test_summary_distinguishes_pointing_miss_from_overlapping_patch(notebook_math):
    maps = {
        "cosine": np.array([[0.5, 0.0], [0.0, 0.0]]),
        "l2": np.array([[1.0, 0.0], [0.0, 0.0]]),
    }
    rows = notebook_math["summarize_pair_maps"](
        {"sample_id": "regression", "label": "regression"},
        maps, np.ones((2, 2), dtype=bool), [100, 200, 10, 3], [100, 200, 128, 228],
    )

    assert rows[0]["point_original_xy"] == [107, 207]
    assert rows[0]["pointing_hit"] is False
    assert rows[0]["peak_patch_overlaps_roi"] is True
    assert rows[0]["roi_in_valid_crop_fraction"] == pytest.approx(1)


def test_summary_excludes_padding_and_handles_roi_outside_valid_area(notebook_math):
    maps = {key: np.array([[1.0, 0.0], [0.0, 0.0]]) for key in ("cosine", "l2")}
    rows = notebook_math["summarize_pair_maps"](
        {"sample_id": "benign", "label": "benign"},
        maps, np.array([[False, True], [True, True]]),
        [100, 200, 10, 3], [100, 200, 128, 228],
    )

    assert rows[0]["valid_patches"] == 3
    assert rows[0]["max"] == 0
    assert rows[0]["point_original_xy"] is None
    assert rows[0]["matched_roi_mean"] is None
    assert rows[0]["roi_in_valid_crop_fraction"] == 0


def test_spatial_tokens_remove_cls_and_registers(notebook_math):
    hidden = torch.arange(18).reshape(1, 9, 2)
    patches = notebook_math["spatial_tokens"](hidden, grid_size=2, register_tokens=4)

    assert tuple(patches.shape) == (1, 4, 2)
    assert torch.equal(patches, hidden[:, 5:, :])
    with pytest.raises(ValueError, match="Expected"):
        notebook_math["spatial_tokens"](hidden, grid_size=2, register_tokens=0)


def test_distance_maps_support_an_explicit_grid(notebook_math):
    features = torch.ones((1, 9, 2))
    maps = notebook_math["distance_maps"](features, features, grid_size=3)

    assert maps["l2"].shape == (3, 3)
    assert maps["l2"].max() == 0
    with pytest.raises(ValueError, match="grid_size squared"):
        notebook_math["distance_maps"](features, features)


def test_summary_uses_explicit_dinov3_patch_coordinates(notebook_math):
    maps = {key: np.array([[1.0, 0.0], [0.0, 0.0]]) for key in ("cosine", "l2")}
    rows = notebook_math["summarize_pair_maps"](
        {"sample_id": "regression", "label": "regression"},
        maps, np.ones((2, 2), dtype=bool), [100, 200, 12, 4], [100, 200, 132, 232],
        grid_size=2, patch_size=16,
    )

    assert rows[0]["point_original_xy"] == [108, 208]
    assert rows[0]["pointing_hit"] is False
    assert rows[0]["peak_patch_overlaps_roi"] is True
    assert rows[0]["roi_in_valid_crop_fraction"] == pytest.approx(1)


def test_dinov3_access_guard_stops_before_loading_weights(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "get_token", lambda: None)
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks/dino_exploration.ipynb").read_text(encoding="utf-8")
    )
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "dinov3-load")

    with pytest.raises(RuntimeError, match="approved model access"):
        exec(compile("".join(cell["source"]), "dinov3-access-guard", "exec"),
             {"OUTPUT_ROOT": tmp_path})


def test_dinov3_processor_matches_the_controlled_dinov2_transform():
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    from PIL import Image
    from transformers import BitImageProcessor, DINOv3ViTImageProcessorFast

    settings = {
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_resize": False,
        "do_center_crop": False,
    }
    pixels = (np.arange(448 * 448 * 3, dtype=np.uint32) % 256).astype(np.uint8)
    image = Image.fromarray(pixels.reshape(448, 448, 3))
    d2 = BitImageProcessor(**settings)(image, return_tensors="pt")["pixel_values"]
    d3 = DINOv3ViTImageProcessorFast(**settings)(image, return_tensors="pt")["pixel_values"]

    assert tuple(d3.shape) == (1, 3, 448, 448)
    torch.testing.assert_close(d3, d2, rtol=0, atol=1e-6)
