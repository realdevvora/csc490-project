"""Fixed, annotation-blind box extraction from nonnegative DINO heatmaps."""
import math

import numpy as np
from scipy import ndimage


BOX_RULE = "half_peak_largest_4connected_v1"


def heatmap_box(values, valid, width, height, cell_px=2):
    """Largest component >= half the valid peak; zero maps use one valid cell.

    Ties use scipy's row-major component order. Boxes use original-image pixels
    and are clipped to the union canvas. No labels or target boxes are inputs.
    A fallback is a localization proposal, not evidence of a detected regression.
    """
    values, valid = np.asarray(values), np.asarray(valid)
    if min(width, height, cell_px) <= 0:
        raise ValueError("Positive image dimensions and cell size required")
    expected = (math.ceil(height/cell_px), math.ceil(width/cell_px))
    if values.shape != expected or valid.shape != expected or valid.dtype != bool:
        raise ValueError("Map/boolean validity mask does not match image geometry")
    if not valid.any() or not np.isfinite(values[valid]).all() or (values[valid] < 0).any():
        raise ValueError("Expected finite nonnegative values on nonempty valid area")
    peak = float(values[valid].max())
    threshold = peak * .5
    foreground = valid & (values >= threshold) if peak > 0 else np.zeros(expected, bool)
    components, count = ndimage.label(foreground)  # 4-connectivity, no padding cells
    fallback = count == 0
    if fallback:
        y, x = np.unravel_index(np.argmax(np.where(valid, values, -np.inf)), expected)
        left, top, right, bottom = x, y, x+1, y+1
    else:
        sizes = np.bincount(components.ravel())[1:]
        ys, xs = ndimage.find_objects(components)[int(np.argmax(sizes))]
        left, top, right, bottom = xs.start, ys.start, xs.stop, ys.stop
    x, y = left*cell_px, top*cell_px
    box = [float(x), float(y), float(min(width, right*cell_px)-x),
           float(min(height, bottom*cell_px)-y)]
    return box, dict(rule=BOX_RULE, peak=peak, map_threshold=threshold,
                     components=count, zero_map_fallback=fallback)
