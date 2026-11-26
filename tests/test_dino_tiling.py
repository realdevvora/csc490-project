import numpy as np
import pytest

from regression_lens.evaluation.tiling import window_origins, add_patch_map, summarize_map


@pytest.mark.parametrize("width,height", [(1, 1), (447, 449), (1001, 1373)])
@pytest.mark.parametrize("patch", [14, 16])
def test_edge_padding_covers_image_and_preserves_patch_alignment(width, height, patch):
    shape = ((height+1)//2, (width+1)//2)
    total, count = np.zeros(shape), np.zeros(shape)
    for x, y in window_origins(width, height):
        assert x % patch == y % patch == 0
        add_patch_map(total, count, np.ones((448//patch, 448//patch)), x, y, patch)
    assert np.all(count > 0)
    np.testing.assert_allclose(total/count, 1)


def test_overlap_is_averaged_in_original_coordinates():
    total, count = np.zeros((8, 12)), np.zeros((8, 12))
    add_patch_map(total, count, np.ones((1, 1)), 0, 0, 16)
    add_patch_map(total, count, np.full((1, 1), 3), 8, 0, 16)
    values = total/count
    np.testing.assert_allclose(values[:, :4], 1)
    np.testing.assert_allclose(values[:, 4:8], 2)
    np.testing.assert_allclose(values[:, 8:], 3)
    assert summarize_map(values, count > 0)["point"] == [17., 1.]


def test_masked_padding_cannot_select_peak_or_inflate_score():
    values = np.array([[.2, .7, 99.]])
    result = summarize_map(values, np.array([[True, True, False]]))
    assert result["max"] == .7
    assert result["top5_mean"] == .7
    assert result["point"] == [3., 1.]


def test_no_coverage_gaps_allowed():
    with pytest.raises(ValueError):
        window_origins(1000, 1000, stride=500)
