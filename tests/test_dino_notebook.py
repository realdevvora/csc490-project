"""Check notebook distance/coordinate math without downloading or loading DINO."""

import ast
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


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


def test_roi_mean_respects_coverage_padding_and_empty_regions(notebook_math):
    values = np.array([[2.0, 8.0], [20.0, 80.0]])
    coverage = np.array([[0.5, 0.5], [0.0, 0.0]])
    valid = np.array([[True, False], [True, True]])

    weighted_mean = notebook_math["weighted_mean"]
    assert weighted_mean(values, coverage) == 5
    assert weighted_mean(values, coverage * valid) == 2
    assert weighted_mean(values, np.zeros_like(coverage)) is None
