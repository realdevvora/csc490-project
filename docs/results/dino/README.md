# DINO dataset-fit pilot

Latest: [reported full-page A/B/C validation results](trained-validation-results.md).

For a student-friendly explanation of A/B/C, the training logs, run counts and
evaluation metrics, see [Understanding the DINO experiments](../../dino-experiments-explained.md).

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

## 2026-11-14: fixed 13-operator pilot and a logged crop follow-up

Checkpoint 3 ran all 39 baseline pair/crop cases locally in approximately 218
seconds (52 forwards, including one reference per trio). The model, seed,
normalization, crop size, and original 13-trio manifest are unchanged. Artifacts
are under ignored `outputs/dino_fit/dinov2/pilot_13/`, including 78 raw distance
summaries, per-trio native maps, a CSV review table, frozen plotting limits, and
an overview. The notebook retains the user's earlier execution outputs.

The following values are **top-5%-mean cosine distances**, not probabilities or
classification metrics. Each row is one example from a different webpage, with
its own magnitude and matched browser control. They cannot establish an
operator-wide ordering of difficulty.

| Operator | Regression top-5% mean | Benign top-5% mean | Peak/geometry observation |
|---|---:|---:|---|
| background_color | 0.2214 | 0.5012 | Centre inside ROI |
| border_radius | 0.04774 | 0.4666 | Centre inside ROI |
| border_width | 0.1648 | 0.3288 | Peak patch outside ROI |
| font_size | 0.03864 | 0.5432 | Patch overlaps ROI; centre misses |
| font_weight | 0.08697 | 0.6948 | Centre inside; crop covers 61% of ROI |
| letter_spacing | 0.01992 | 0.5740 | Patch overlaps ROI; centre misses |
| line_height | 0.02086 | 0.4654 | Peak patch outside ROI |
| margin | 0.1324 | 0.3426 | Centre inside; crop covers 98% of ROI |
| opacity | 0.01755 | 0.7144 | Centre inside ROI |
| padding | 0.05588 | 0.5447 | Overlapping patch only; 75% ROI coverage |
| position, original crop | ~1.83e-7 | 0.6055 | ROI outside crop: not evaluable for sensitivity |
| text_color | 0.0003714 | 0.3803 | Very weak response; overlapping patch only |
| width | 0.5957 | 0.4216 | Strong response; centre inside; 77% ROI coverage |

### Interpretation

- **Clearest selected case:** the width mutation changes text wrapping and
  produces a strong spatial response. It is the only original case whose
  top-5%-mean distance exceeds its matched benign control's value. This does not
  make all width changes easy, and its large changed region helps its pointing
  result.
- **Weak selected case:** the small text-color mutation has a maximum cosine
  distance of only 0.001341, versus a matched benign top-5% mean of 0.3803. It is a
  useful difficult example for the later DINOv3 comparison.
- **Local response versus nuisance:** several regression responses concentrate
  near their targets, but 12 of the 13 original browser controls have larger
  top-5%-mean distances than their regressions. One of those 13 regression crops
  is uninformative, as explained below. Raw distance magnitude alone is not a
  reliable semantic defect score on these examples.
- **Localization resolution:** six original cosine-map peaks have their centres
  inside the shared target ROI. Four more peaks have overlapping patches but
  centres outside; two peaks are elsewhere, and position has no valid ROI in its
  crop. Keep these outcomes distinct rather than counting overlap as a pointing
  hit. Large-box hits are less informative than small-box hits.
- **Identity:** all 13 original no-change L2 maps have maximum distance exactly
  zero. Cosine differences remain at float32 roundoff.

### Position follow-up: input coverage, not a failed representation

The original position crop `[426, 1876, 874, 2324]` is centred on the union of a
983 px-wide DOM element before/after translation. The actual changed glyphs are
near its left edge: `diff_bbox = [148, 2226, 61, 11]`. They are completely outside
that crop. Its near-zero feature map therefore cannot assess position sensitivity.

The original manifest and baseline outputs are retained. A **separate**
`position_supplement_manifest.json` records the same three sample IDs with crop
`[0, 1876, 448, 2324]`, centred on the known changed region. This adds three
follow-up pair/crop cases, for 42 evaluations of 39 unique pairs. It changes no
source data, labels, exclusions, or team splits.

With this crop, regression top-5%-mean cosine distance is 0.08048, versus 0.4697
for the matched Firefox control. Mean response inside the ROI is 0.1885 versus
0.006595 outside. The peak patch overlaps the translated text, while its centre
is just outside the tight box. No-change L2 remains zero. The response was
previously absent because the changed content was not visible in the input.

Repeat both the original manifest and the supplementary position manifest with
DINOv3. Do not silently replace the original crop or mix the two measurements in
a single baseline row. The same source page can place the visible change far
from its DOM-box centre; crop coverage must be considered when interpreting a
representation study.

### Presentation and next step

Use the width figure, weak text-color figure, and position follow-up to explain
spatial sensitivity, nuisance response, and input-coverage limitations. Compare
DINOv3 on these frozen inputs next. More pages per operator and eventual
page-level inference are needed before making generalization or defect-detection
claims; neither head training nor DiffSpot tuning is part of this pilot.

## 2026-11-14: DINOv3 access check and prepared comparison

The official `facebook/dinov3-vitb16-pretrain-lvd1689m` checkpoint reports manual
access gating. An anonymous configuration-file request for revision
`5931719e67bbdb9737e363e781fb0c67687896bc` returned HTTP 401 / `GatedRepoError`.
At that initial check this computer had no Hugging Face login configured.
Access and login subsequently succeeded; see the executed results below.

Checkpoint 4 is prepared to repeat the font-size trio, remove the configured
register tokens as well as CLS, assert a 28 x 28 patch grid, and compare with
saved DINOv2 maps on the same crop/normalization/display scale. Unit tests cover
token separation, explicit grid/coordinate math, and the missing-login guard.
Those synthetic checks are infrastructure verification, not pretrained results.
Recomputing the saved DINOv2 summaries with the explicit-grid helpers leaves all
78 summaries unchanged. The DINO notebook test group passes all 10 cases,
including preprocessing equivalence with resizing/cropping disabled.
See the [notebook guide](../../../notebooks/README.md) for the official access
page and local login command. After the actual first-trio comparison works,
repeat both frozen manifests before drawing a backbone-level conclusion.

## 2026-11-14: executed DINOv3 first-trio comparison

Checkpoint 4 completed locally on CPU with the pinned official weights,
float32, disabled resizing/centre cropping, and the same font-size trio and
normalization as DINOv2. The sequence has 789 tokens: CLS, four registers, and
784 spatial tokens (28 x 28). All parameters remain frozen.

Top-5%-mean cosine distances on identical crops:

| Pair | DINOv2 | DINOv3 |
| --- | ---: | ---: |
| Font-size regression | 0.0386424 | 0.0261631 |
| Benign WebKit | 0.543158 | 0.116949 |
| No change | 1.983e-7 | 2.041e-7 |

DINOv3 has a smaller benign response, but its regression response also decreases.
The benign/regression ratio is about 4.47 versus 14.06 for DINOv2 on this one
trio; benign still exceeds regression. Raw distance scales need not be calibrated
across backbones, so this does not establish better detection accuracy.

The DINOv3 regression response is concentrated near the changed text, but the
maximum patch neither overlaps the tight 54 x 9 px pixel-change ROI nor has its
centre inside it. ROI mean cosine distance is 0.04805 versus 0.001760 outside.
No-change L2 distances are exactly zero; cosine residuals are float32 roundoff.
The shared-scale comparison figure was inspected. Native maps, statistics, and
the figure are under ignored `outputs/dino_fit/dinov3/font_size/`.

Next repeat all 39 original pair/crop cases and the separate three-case position
follow-up with DINOv3 on the lab GPU. Do not infer operator-wide performance from
this single font-size example. No head training or full-page evaluation has run.

## 2026-11-14: completed matched DINOv3 pilot

The [standalone runner](matched-pilot-run.md) completed all 39 frozen baseline
pair/crop cases and the separate three-case position supplement on local CPU.
It saved 84 raw distance summaries, 14 native-map archives and 14 shared-scale
comparison figures in `outputs/dino_fit/dinov3/pilot_13/`. Recorded image-forward
time totals 207.4 seconds, excluding preprocessing, plotting and model loading.
The repeated font-size native maps match the notebook artifacts exactly.

Selected DINOv3 cosine results (top-5%-mean):

| Mutation example | Regression | Benign | Strict peak inside ROI |
| --- | ---: | ---: | --- |
| background_color | 0.058161 | 0.196779 | No; patch overlaps |
| border_radius | 0.030281 | 0.077097 | Yes |
| border_width | 0.082481 | 0.022916 | Yes |
| font_size | 0.026163 | 0.116949 | No; patch does not overlap |
| font_weight | 0.056021 | 0.208328 | Yes |
| letter_spacing | 0.020988 | 0.269381 | No; patch overlaps |
| line_height | 0.001802 | 0.183054 | No; patch overlaps |
| margin | 0.050876 | 0.113136 | Yes |
| opacity | 0.006575 | 0.287839 | Yes |
| padding | 0.029332 | 0.220058 | Yes |
| position, original crop | ~2e-7 | 0.286393 | Not assessable: ROI absent |
| text_color | 0.000283 | 0.043235 | No; patch overlaps |
| width | 0.342278 | 0.118551 | Yes |
| position, separate supplement | 0.030771 | 0.182102 | Yes |

On the original 13 trios, benign exceeds regression in 11 DINOv3 examples
versus 12 DINOv2 examples. Width and border-width are the two DINOv3 examples
where regression exceeds benign. Small text-color and line-height changes have
weak regression responses, while opacity localizes but has poor separation from
its browser control. These describe the chosen instances, not all mutations of
each type. Seven baseline peaks are strictly inside the ROI, four have only
patch overlap, one is elsewhere, and position has no ROI coverage. Partial ROI
coverage and large target boxes limit interpretation of these counts.

In the separate position crop, DINOv3's peak is inside the tight ROI; DINOv2's
peak only overlaps it. Benign still dominates both. All 14 no-change L2 maxima
are exactly zero. The width comparison figure was inspected after the completed
run, in addition to the earlier font-size figure. There is no held-out accuracy
or trained-head result yet. Next engineering work is label-blind full-page
coverage on the lab GPU, followed by the authorized frozen spatial-head and
benign-data ablation experiments.
