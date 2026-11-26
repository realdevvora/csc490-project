# ADR 0006: Add heatmap-derived boxes as a separate DINO validation extension

- **Status:** Accepted
- **Date:** 2026-11-16
- **Owners:** DINO investigation track
- **Related:** [Procedure and commands](../results/dino/box-localization.md)

## Context

The completed DINO A/B/C runs export scores and points, so the shared scorer
cannot compute box IoU. The user authorized adding boxes without retraining.
Saved full-page heatmaps already provide the necessary spatial output. The
local report bundle excludes those maps; the lab originals are required.

## Decision

Preserve the original v1 predictions and extend them in a separate output
namespace. Apply `half_peak_largest_4connected_v1`: threshold each valid map
at half its maximum, box the largest four-connected component, resolve equal
sizes in row-major order, and use the first valid peak cell for zero maps.
Clip original-coordinate boxes to the union canvas. Apply this rule to every
pair without consulting labels, target boxes or pixel differences.

Reuse original validation-selected A/B/C checkpoints and existing score/point
outputs. Do not tune box thresholds or checkpoints against IoU. Score with
the unchanged shared scorer and aggregate IoU per seed, reporting sample SD.
Call this a post-hoc validation localization extension. Keep type metrics
unavailable; adding a type classifier is outside this change.

## Rationale and alternatives

A fixed relative threshold supports both distance maps and learned maps
without assuming comparable calibration. Threshold search on the same
validation set would add selection optimism. A single patch box or whole
window box is easier but discards spatial extent. Pixel-based refinement
would add another evidence source and change what is being measured.

## Consequences

No model training, GPU or split change is required. Raw maps must be present;
otherwise rerun inference, not training. Diffuse maps can produce large boxes
and disconnected changes can be missed. Zero-map fallbacks are proposals,
not detections. IoU is against diff_bbox even for reflow, while pointing uses
bbox_after there. Report both with these distinct meanings and box coverage.

## Validation and revisit

Tests cover padding, odd-coordinate clipping, connectivity/ties, zero maps,
bad-map rejection, identity mapping and score/point preservation. Numerical
dataset IoU awaits processing of lab maps. Any future learned/calibrated box
procedure should be a separately named experiment with documented selection.

Number 0005 was already used for the team's DiffSpot protocol on the CLIP
investigation branch at creation; it is now integrated from main. This
decision does not change that protocol.
