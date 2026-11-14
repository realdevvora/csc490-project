# DINOv2 / DINOv3: findings and lab experiment controls

Recorded 2026-11-14 after the matched pretrained feature pilot. This document
separates observed limitations from controls for the upcoming experiments.
It does not change dataset roles, labels, or the committed source-page split.
General dataset validation and manual inspection are complete and are not being
repeated. Evidence and raw-artifact locations are in [the pilot notes](README.md).

## Findings to carry forward

| Finding | Evidence | Implication |
| --- | --- | --- |
| Raw distance is not yet a semantic regression score. | Benign browser variation exceeds regression top-5%-mean cosine distance in 12/13 DINOv2 and 11/13 DINOv3 baseline trios. | Expect false positives from a simple distance threshold; evaluate benign rejection explicitly. |
| Smaller DINOv3 distances do not establish better performance. | Both regression and benign responses decrease in the first font-size comparison. | Calibrate each model separately using training/validation data; compare downstream metrics rather than distance magnitudes. |
| Crop coverage can dominate apparent model sensitivity. | The original position crop excludes the changed glyphs; the separate supplement reveals a response. | Full-page window selection must not consult mutation locations. Check spatial coverage and coordinate mapping before interpreting failures. |
| Small changes can produce weak responses. | The selected text-color and line-height examples have weak DINOv3 responses; text-color was also weak with DINOv2. | Retain native screenshot scale initially and report results by operator; these examples do not establish operator-wide difficulty. |
| The patch grids differ. | At 448 px, DINOv2-B/14 produces 32 x 32 spatial tokens and DINOv3-B/16 produces 28 x 28. | Preserve each native grid and map predictions into original screenshot coordinates. A common input size does not give equal localization resolution. |
| Localization and detection measure different behavior. | Responses may concentrate near a mutation while benign controls still have larger scores. | Report detection and localization separately; distinguish strict pointing hits from patch overlap and note ROI size/coverage. |
| The pilot cannot establish generalization. | There is one selected regression per operator, on training pages, with mutation-guided crops. | Do not report pilot counts as held-out accuracy or a definitive backbone ranking. |
| Identity controls currently behave correctly. | All no-change L2 maxima are exactly zero; cosine residuals are float32 roundoff. | Retain these controls as an inference sanity check. |

## Controls for the upcoming lab experiments

- Use the same source images, committed page-disjoint split, window-selection
  policy, preprocessing rules and evaluation protocol for both backbones.
  Keep all derivatives of a source page together. Do not use ground-truth boxes
  to select inference windows; boxes remain available for supervision and scoring.
- Start with full-page coverage and original-coordinate mapping, then frozen
  backbones with the same small spatial-head design. The differing native grids
  must be handled explicitly. No VLM is part of this head; VLM integration is a
  later group stage.
- Compare regression + no-change training against regression + no-change + benign
  training. Evaluate both on the same validation set containing all three labels.
  Record sampling/class balance and training steps so extra benign data does not
  silently change the optimization budget.
- Choose thresholds and tune settings using the committed training/validation
  split only. Report validation results as validation results, not independent
  test performance. Keep DiffSpot untouched for external evaluation after system
  selection, under the existing ADRs.
- Record pinned model revisions, processor settings, frozen/trainable parameter
  counts, seed, head design, training budget, optimizer settings, class balance,
  software environment, peak GPU memory and throughput. Use matched settings
  where meaningful and document unavoidable differences between backbones.
- Include a cheap pixel-difference baseline under the same inference/evaluation
  rules to assess what DINO features add. Do not tune it on DiffSpot.
- Preserve raw scores and maps alongside metrics; use shared, documented display
  scales rather than independently normalizing every heatmap.

## Immediate sequence and remaining decisions

1. Implement label-blind full-page coverage and verify coordinate mapping on the
   lab GPU. This is model-inference verification, not another dataset review.
2. Run full-page pretrained baselines for both backbones under matching rules.
3. Train the frozen spatial heads and the benign-data ablation; compare detection,
   localization, resource usage and failure examples on the shared validation set.
4. Repeat with additional seeds as the schedule permits before making robustness
   claims; any later backbone fine-tuning is a separate condition.

Window stride/overlap, padded-border handling, aggregation of overlapping patch
scores, head architecture, training budget and threshold-selection objective
still need concrete implementation choices. Record those choices before large
runs; do not treat them as already implemented. No full-page or trained-head
results exist at the time of this note.
