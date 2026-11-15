"""Frozen paired-patch head and weak box supervision (no segmentation claims)."""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def paired_features(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError("Expected matching BCHW patch grids")
    reference, candidate = F.normalize(reference, dim=1), F.normalize(candidate, dim=1)
    return torch.cat(((reference-candidate).abs(), reference*candidate,
                      (reference*candidate).sum(1, keepdim=True)), 1)


class SpatialHead(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(channels*2+1, 128, 1), nn.GELU(),
                                   nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Dropout(.1))
        self.location = nn.Conv2d(64, 1, 1)
        self.detection = nn.Linear(128, 1)

    def forward(self, reference, candidate, valid=None):
        x = self.trunk(paired_features(reference, candidate))
        if valid is None:
            valid = torch.ones_like(x[:, 0])
        weights = valid[:, None].to(x.dtype)
        mean = (x*weights).sum((2, 3))/weights.sum((2, 3)).clamp_min(1)
        maximum = x.masked_fill(weights == 0, -1e4).amax((2, 3))
        return self.detection(torch.cat((mean, maximum), 1)).squeeze(1), self.location(x)[:, 0]


def box_overlap(box, crop, patch, grid):
    x, y, w, h = box
    xs = np.arange(grid)*patch+crop[0]
    ys = np.arange(grid)*patch+crop[1]
    return np.outer(np.maximum(0, np.minimum(ys+patch, y+h)-np.maximum(ys, y)),
                    np.maximum(0, np.minimum(xs+patch, x+w)-np.maximum(xs, x)))/patch**2


def weak_targets(record, crop, patch):
    """Union validity; intersecting patches are weak foreground, not pixel masks."""
    grid = 448//patch
    valid = np.zeros((grid, grid), np.float32)
    for w, h in (record['size_ref'], record['size_cand']):
        valid = np.maximum(valid, box_overlap([0, 0, w, h], crop, patch, grid)).astype(np.float32)
    target, mask = np.zeros_like(valid), valid.copy()
    changed = 0.
    if record['label'] == 'regression':
        diff = box_overlap(record['diff_bbox'], crop, patch, grid)
        changed = float(np.any((diff > 0) & (valid > 0)))
        box = record['bbox_after'] if record['reflow'] else record['diff_bbox']
        target = (box_overlap(box, crop, patch, grid) > 0).astype(np.float32)
        if record['reflow']:
            mask[(diff > 0) & (target == 0)] = 0
    return dict(valid=torch.from_numpy(valid), target=torch.from_numpy(target),
                mask=torch.from_numpy(mask), changed=torch.tensor(changed))


def balanced_location_loss(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    # Balance foreground/background separately for each window, not across a batch.
    positive, negative = mask*target, mask*(1-target)
    terms, present = [], []
    for weights in (positive, negative):
        count = weights.sum((1, 2))
        terms.append((loss*weights).sum((1, 2))/count.clamp_min(1e-12))
        present.append((count > 0).to(loss.dtype))
    active = present[0]+present[1]
    per_window = (terms[0]*present[0]+terms[1]*present[1])/active.clamp_min(1)
    return per_window[active > 0].mean() if (active > 0).any() else logits.sum()*0


def head_loss(outputs, batch):
    detection, location = outputs
    det = F.binary_cross_entropy_with_logits(detection, batch['changed'])
    loc = balanced_location_loss(location, batch['target'], batch['mask'])
    return det+loc, det, loc
