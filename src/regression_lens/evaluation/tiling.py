"""Label-blind native-scale windows and overlap averaging in image coordinates."""
import math

import numpy as np


def window_origins(width, height, size=448, stride=224):
    if min(width, height, size, stride) <= 0 or stride > size:
        raise ValueError("Positive dimensions and stride <= window size required.")
    # Keep the regular grid at the far edge; pad rather than moving the last tile.
    return [(x, y) for y in range(0, max(1, math.ceil((height-size)/stride)+1)*stride, stride)
            for x in range(0, max(1, math.ceil((width-size)/stride)+1)*stride, stride)]


def add_patch_map(total, count, values, x, y, patch_size, scale=2):
    if patch_size % scale or x % scale or y % scale:
        raise ValueError("Patch and window origins must align with output grid.")
    expanded = values.repeat(patch_size//scale, 0).repeat(patch_size//scale, 1)
    x, y = x//scale, y//scale
    h, w = min(expanded.shape[0], total.shape[0]-y), min(expanded.shape[1], total.shape[1]-x)
    total[y:y+h, x:x+w] += expanded[:h, :w]
    count[y:y+h, x:x+w] += 1


def summarize_map(values, valid, scale=2):
    raw = values[valid]
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Expected finite distances on nonempty image area.")
    peak = int(np.argmax(np.where(valid, values, -np.inf)))
    y, x = np.unravel_index(peak, values.shape)
    return dict(mean=float(raw.mean()), max=float(raw.max()),
                top5_mean=float(np.sort(raw)[-max(1, math.ceil(.05*len(raw))):].mean()),
                point=[float(x*scale+scale/2), float(y*scale+scale/2)])
