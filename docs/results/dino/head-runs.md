# Frozen-head experiments B and C

Implements the accepted six-configuration plan. DiffSpot, backbone fine-tuning,
operator classification and VLM integration are excluded. The committed 390/69
source-page split and exclusions are unchanged; no extra holdout is introduced.
General dataset inspection is not repeated.

## Protocol

**Crop audit / correction:** A later geometry audit of the frozen v1 manifest
found 3,263 fully covered regression anchors, 1,474 partially covered anchors,
and two zero-coverage anchors out of 4,739. The two are `d2c_314_margin` and
`d2c_329_position`: changed strips extend beyond the reference dimensions.
Their zero-change-window targets exclude them from positive training sampling;
they are not incorrect positive labels. Geometric intersection does not prove
that a crop contains changed pixels within a large bounding box; preview checks
remain necessary. Partial coverage is expected for targets larger than 448 px.

The v2 correction preserves every already-covered anchor and uses union-of-image
extents only when reference-clamping misses the scoring target or changed box.
Controls retain their anchor's exact crop coordinates. On the actual manifest,
only three crops change: these two regression anchors and `d2c_329_webkit`'s
matched anchor. All 4,739 corrected regression anchors intersect their target
and changed box. This is crop-specific verification, not repeated dataset review.

The user confirmed B/C v1 jobs are already running. Preserve those jobs, manifests,
caches and results unchanged. No experiment A rerun is needed. The local corrected
launcher uses fresh `outputs/dino_heads_v2/` paths; it is for a deliberately
chosen corrected run after the current experiment, not a mid-run replacement.
Do not combine v1 and v2 heads within a controlled comparison. A rerun is not
required solely by these two omitted anchors; disclose the limitation in v1
results and assess the completed runs first.

`scripts/dino_heads.py` provides `manifest`, `extract`, `train`, `evaluate` and
`report` subcommands. All support `--help`. Global `--system-trust` enables the
optional local Windows certificate-store workaround and goes before the subcommand.

The model-independent manifest freezes two windows per training pair, seed 490:
one uniform origin shared across same-page variants, and one scorer-target-centred
origin. Benign/no-change anchors use a stable hash to select a regression on the
same page. Original coordinates, source IDs, metadata and split hashes are saved.
Ground truth is allowed for training crops only. A regression window is positive
when it intersects `diff_bbox`; background-only regression windows are not sampled
as classification negatives. Benign/no-change windows supply those negatives.

Extraction uses the same pinned models, normalization and native 448 px geometry
as experiment A. Frozen float32 feature pairs and weak targets are cached once
for all six trained runs of a backbone. Storage is estimated before extraction:
96 GiB maximum per backbone, with at least 120% of estimated bytes free at start.
There is no automatic eviction, precision downgrade or sampling fallback. Cache
files are ignored local artifacts; budget for both backbones plus evaluation maps.
The actual committed training subset yields 11,818 sampled windows across all
390 pages. Estimated float32 cache sizes are 69.34 GiB for DINOv2 and 53.11 GiB
for DINOv3 (approximately 122.45 GiB combined, before validation maps).

The head individually L2-normalizes reference/candidate patch vectors and uses
absolute differences, elementwise products and cosine similarity (1,537 channels).
Its trunk is 1x1 convolution to 128, GELU, 3x3 convolution to 64, GELU, dropout .1.
One branch produces localization logits; another classifies each window from
validity-weighted mean and masked-maximum pooled trunk features.

Training uses AdamW, LR .0003, weight decay .01, batch 32 and 2,000 updates, with
checkpoints at 500/1,000/2,000. No augmentation or learning-rate search is applied.
Each step has 16 regression-positive windows. B adds eight benign and eight
no-change windows; C adds 16 no-change windows. Identical seeded positive streams
and identical update counts isolate the effect of negative supervision.

Loss is detection BCE plus balanced weak-localization BCE, weight one each.
Foreground patches intersect `diff_bbox`, or `bbox_after` for reflow. In reflow,
displaced patches within `diff_bbox` but outside the target are ignored for
localization. Union-of-image validity weights remove padded-only loss regions.
Boxes supply weak targets, not pixel masks. Tiny boxes can label an entire patch.

Seeds are 490/491/492 with deterministic PyTorch algorithms. Training diagnostics
record loss components, gradient norm and loss on a fixed common training sample
that excludes benign pairs in both conditions. C never uses benign targets for
optimization or these diagnostics. Validation necessarily contains benign pairs.
The fixed-training AUROC is window-level, while reported validation AUROC is
full-page; do not interpret their numerical difference as a pure generalization
gap. Training-source diagnostics, losses and full-page error review must be read
together. Seed standard deviation measures training variability, not a confidence
interval for unseen source pages.

Full-page validation streams each window's backbone features through all 18 heads
(two conditions x three seeds x three checkpoints), without a full-page feature
cache. Inference reads image dimensions only for window geometry. Localization
probabilities are averaged on the existing 2 px coordinate grid; pair confidence
is the highest window confidence. Original-coordinate peak points are written;
no predicted boxes or mutation types are fabricated. This differs from A's raw
distance score, but the B/C policy is identical across both backbones.

Each head gets prediction JSONL, raw localization maps, shared metrics, operator
and fixed change-size breakdowns. Size bands use `diff_bbox` area/reference image
area: <=0.1%, <=1%, and >1%. Shared scoring also reports magnitude/reflow results.
Checkpoints must belong to completed runs and match dataset/split/backbone config.

`scripts/review_dino_heads.py` produces training-loss, fixed-training-loss,
gradient and validation-AUROC plots, plus seed-490 examples for each selected
condition: the two highest-scoring benign pairs and two lowest-scoring regression
pairs. Each figure states whether it is an actual thresholded error; difficult
correct predictions are not mislabeled as failures. Review uses shared 0–1
localization probability scales and adds known targets only after inference.

Select a common checkpoint step per condition by three-seed mean validation
regression/benign AUROC, then lower mean benign FPR@95R, then earlier step.
Thresholds maximize validation F1 separately for each head. Report all seeds and
means/sample standard deviations. These are validation-selected results, not an
independent test. Record any recipe revision as a new experiment.

## Launch on the lab machine

Develop/test locally while experiment A runs. Do not launch the GPU stages until
both existing validation jobs finish. Transfer the latest source archive, then:

```bash
export LAB_ROOT="/virtual/.cache/$USER/csc490"
cd "$LAB_ROOT/csc490-project"
nohup bash scripts/run_dino_head_experiments.sh > "$LAB_ROOT/dino-heads.log" 2>&1 &
tail -f "$LAB_ROOT/dino-heads.log"
```

The runner checks both baseline completion markers, then runs one backbone at a
time, six training runs per backbone, full-page evaluation and reporting. It uses
the lab environment's `python3` throughout. Existing completed stages are skipped;
an incomplete nonempty stage fails rather than silently overwriting artifacts.
Preserve an incomplete stage separately before restarting that stage. Do not edit
or replace running source/cache artifacts.

Corrected v2 combined results are under `outputs/dino_heads_v2/final_comparison/`, alongside
the baseline and per-backbone validation reports. The combiner requires the full
1,033-pair validation set. This is a single bounded recipe, not a guarantee of no
overfitting/underfitting. Inspect diagnostic curves and validation errors before
making representation or robustness claims.

## Small engineering smoke run

Use a separate manifest/output tree with `manifest --limit-pages 1`, `train
--steps 2` and `evaluate --limit-pages 1`. Smoke results are marked and cannot be
used by `report` to select final checkpoints. The synthetic overfit unit test is
an optimization check, not dataset evidence. No lab GPU timing or head accuracy
claim should be made until the new jobs have actually run.

## Verification completed locally

The test suite covers target geometry/reflow masking, padding validity, symmetric
paired features, deterministic manifests, equal B/C regression streams, synthetic
optimization with frozen backbone parameters, checkpoint reload equivalence,
three-seed checkpoint selection/tie rules, and rejection of smoke selection.
Separate synthetic-data CLI smoke runs exercise actual pinned DINOv2 and DINOv3
extraction, two-update head training and full-page scoring. These do not replace
the real-data lab runs. The shell launcher has passed Bash syntax checking.
