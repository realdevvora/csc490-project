# DINO dataset-fit pilot

## 2026-11-14: one font-size trio, frozen DINOv2

This is a qualitative, ground-truth-guided crop study on official training pages.
The team's general dataset validation is treated as completed. No head was trained,
no validation or external benchmark score was selected, and no claim about all
font-size examples follows from this single trio.

### Reproduction

Run checkpoints 1 and 2 of `notebooks/dino_exploration.ipynb` using the environment
in the [notebook guide](../../../notebooks/README.md). The study selects 13 matched
trios with seed 490, but this first inference run evaluates only:

- `d2c_200_font_size` (regression);
- `d2c_200_webkit` (benign); and
- `d2c_200_identical` (no change).

All images use the same original-scale 448 x 448 crop, `[70, 648, 518, 1096]` in
`[left, top, right, bottom]` coordinates. The model is `facebook/dinov2-base` at
revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, with float32 frozen parameters,
SDPA attention, checkpoint normalization, and no processor resize/centre crop.
Its final 1024 spatial tokens have 768 channels and form a 32 x 32 patch grid.

The local run used Python 3.11.4, PyTorch 2.8.0+cpu, Transformers 4.57.6, and four
CPU threads on a Ryzen 7 3700U. Initial forwards took approximately 3.1–3.4 seconds
per image, excluding model download and image preprocessing. That makes further
small-sample local inference practical before moving to a lab GPU.

Local figures, unscaled maps, statistics, environment details, notebook hash,
working-tree state, model revision, and sample-manifest hash are retained under
ignored `outputs/dino_fit/dinov2/font_size/`. Outputs are generated from notebook
code before committing; the run record identifies that dirty working tree.

### Observations

| Pair | Mean cosine distance | Maximum cosine distance | Mean raw L2 distance |
|---|---:|---:|---:|
| Font-size regression | 0.002107 | 0.470874 | 1.12389 |
| Benign WebKit render | 0.125935 | 0.657792 | 22.5861 |
| No change | 3.57e-8 | 2.98e-7 | 0 |

- **Local sensitivity:** the regression heatmap concentrates near the edited
  text. Mean cosine distance weighted by overlap with its changed-region box is
  approximately 0.2211, versus 0.001575 outside that region.
- **Nuisance sensitivity:** the benign browser render causes larger, broader
  responses. Raw feature distance alone does not separate the regression from
  browser variation in this example. These are distances, not probabilities.
- **Identity behaviour:** the no-change L2 map is exactly zero. Its tiny cosine
  values are float32 roundoff; independent per-map contrast scaling would make
  that noise misleadingly visible.
- **Patch-resolution limitation:** the highest-distance patch is near the edit,
  but its centre `[287, 879]` lies outside the tight `[238, 867, 54, 9]` changed box.
  The 14 px patch overlaps the box while its centre lies 3 px below the lower
  edge. The shared pointing rule therefore correctly reports a miss. Keep that
  result rather than enlarging ground truth to improve the metric.

The figures use one shared colour scale per distance across all labels. Native
maps supply all statistics; interpolation is only for display. L2 depends on
feature magnitude, so the later backbone comparison should interpret it alongside
cosine rather than assuming raw L2 scales are interchangeable between models.

### Next question

Do the other preselected operators show the same local sensitivity and stronger
benign activation? Evaluate the already frozen sample before forming operator-wide
conclusions, then repeat identical crops with DINOv3 once weight access is granted.
