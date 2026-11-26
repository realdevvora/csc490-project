# Full-page pretrained baseline

`scripts/run_dino_fullpage.py` supports pinned DINOv2-B/14 and DINOv3-B/16
with frozen float32 SDPA inference. Window selection depends only on image
dimensions, never mutation locations or labels. This implements the coverage
stage in [experiment controls](experiment-controls.md); no head is trained.

Local verification: nine geometry tests pass, including both patch sizes,
irregular image edges, overlap averaging and padded-area exclusion. A small
synthetic identity input exercises the pretrained end-to-end path separately
from dataset experiments. A real full-page CPU check was interrupted because of
runtime cost; it is incomplete and provides no dataset result. The first real
throughput and memory check has now completed on the lab GPU, as recorded below.

## Reported RTX 4080 smoke timings

The user reported both completed runs on the same training page (15 pairs):

| Model | Elapsed seconds | Pairs/second | Peak PyTorch allocated bytes |
| --- | ---: | ---: | ---: |
| DINOv2 | 33.194 | 0.4519 | 455449600 |
| DINOv3 | 20.304 | 0.7388 | 414596096 |

DINOv3 took about 39% less elapsed time for this smoke workload. This is one
page, not a controlled multi-page throughput benchmark or an accuracy result.
The allocation measurement excludes CUDA context, display and other processes;
it is not total GPU memory usage. Completion summaries alone do not verify
identity scores or predicted coordinates.

The committed validation split contains 1,033 pairs across 69 pages: 826
regressions, 138 benign controls and 69 no-change controls. Linear extrapolation
from this page is approximately 38 minutes for DINOv2 and 23 minutes for DINOv3,
but page dimensions/window counts and I/O may substantially change runtime.
Full validation metrics have now been reported below; trained-head results remain
pending.

## Reported experiment A validation results

The user supplied the shared scorer's output for all 1,033 validation pairs.
Values below are rounded terminal values, not independently inspected raw files.
Checkpoint weights remain pretrained/frozen and thresholds maximize validation F1.

| Model | AUROC regression/benign | Benign FPR@95R | Precision | Recall | F1 | Benign FPR | No-change FPR | Pointing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 | 0.082 | 1.000 | 0.857 | 1.000 | 0.923 | 1.000 | 0.000 | 0.539 |
| DINOv3 | 0.206 | 1.000 | 0.857 | 1.000 | 0.923 | 1.000 | 0.000 | 0.585 |
| Pixel | 0.097 | 1.000 | 0.857 | 1.000 | 0.923 | 1.000 | 0.000 | 0.953 |

All three selected thresholds flag all 826 regressions and all 138 benign pairs,
while rejecting all 69 no-change pairs. Thus TP=826, FP=138, TN=69, FN=0;
precision is 826/964 and F1 is 1652/1790. The high F1 reflects the class mix and
separation from identity controls, not successful benign rejection. No method
rejects benign pairs while maintaining 95% regression recall in this baseline.

AUROC below 0.5 means benign scores generally exceed regression scores, consistent
with the pilot's nuisance sensitivity. DINOv3 is numerically less poor at this
ranking and has higher pointing accuracy than DINOv2, but neither is a successful
semantic detector here. Seed variation/confidence intervals are not supplied by
these single deterministic baseline runs. Do not claim a definitive backbone win.

Pixel difference is the strongest pointing baseline. For non-reflow cases the
target `diff_bbox` is itself pixel-difference-derived, so this is an expected
geometric advantage; it does not establish semantic detection. Pointing is
reported on regression pairs independently of the chosen detection threshold.
No boxes or operator outputs are produced, so IoU/type metrics are unavailable.

The table shows every threshold as 0.000 because it prints only three decimal
places. Read the saved metrics JSON for exact thresholds; do not infer that they
are exactly zero or equal across models. Preserve the baseline scoring direction
and thresholds. Any score inversion or revised pooling would be a separately
declared experiment, not a replacement chosen to improve these reported numbers.

Next run the already specified B/C conditions unchanged. The main question is
whether benign supervision improves regression/benign ranking and rejection
without sacrificing regression detection/localization. The no-benign ablation
may retain the current nuisance sensitivity; improvement is not guaranteed.

Policy fixed before validation runs:

- Use native-scale 448 x 448 inputs, stride 224, without resizing/centre crop.
  Far-edge windows stay on the regular grid and are padded white. The stride
  aligns with both patch sizes.
- Use the union of reference and candidate extents so size changes remain
  observable. Include patch responses at image boundaries, but exclude output
  cells outside both real images. Padding can influence boundary patch features;
  this is an explicit limitation of the policy.
- Remove CLS and configured register tokens. Compute raw corresponding-patch
  cosine and L2 distances. Project constant patch values onto a common 2 px
  screenshot grid, then average equally across overlapping windows. This grid
  preserves both patch boundaries; it adds no actual feature resolution.
- Score each pair by the mean of its largest 5% valid cosine-map cells. Report
  the maximum cell centre as the original-coordinate point; suppress the point
  for numerical identity maps with maximum cosine <= 1e-6. No box or operator
  prediction is fabricated.
- Also compute a pixel RGB mean-absolute-difference / 255 baseline with the same
  white padding and union mask. Its top-5% score and peak use native pixels;
  the saved strided pixel preview is not used to compute scores.

Outputs: `config.json`, `predictions.jsonl`, `pixel_predictions.jsonl`, raw
`maps/*.npz`, per-pair `statistics.jsonl`, and `complete.json` with total time,
throughput and peak GPU allocated memory. Nonempty output directories are
rejected. Interrupted runs lack `complete.json` and must not be scored as full
validation. There is no resume or reference-feature cache yet; throughput must
be measured on the lab host before predicting run duration.

## Lab smoke test

After transferring the latest working-tree archive, run from the lab repo:

```bash
export LAB_ROOT="/virtual/.cache/$USER/csc490"
export HF_HOME="$LAB_ROOT/huggingface"
export TMPDIR="$LAB_ROOT/tmp"
cd "$LAB_ROOT/csc490-project"
"$LAB_ROOT/.venv/bin/python3" scripts/run_dino_fullpage.py \
  --data-root "$LAB_ROOT/data/d2c_v3" --model dinov2 \
  --subset train --limit-pages 1 --out outputs/fullpage/dinov2_smoke
"$LAB_ROOT/.venv/bin/python3" scripts/run_dino_fullpage.py \
  --data-root "$LAB_ROOT/data/d2c_v3" --model dinov3 \
  --subset train --limit-pages 1 --out outputs/fullpage/dinov3_smoke
```

These use identical sorted source pages and all their pairs. `--limit-pairs N`
is available for a smaller engineering check; neither limited mode is a full
validation result. Review no-change maps, predicted coordinates, pair timings
and memory before scaling up.

## Full validation and shared scoring

For each model, replace `--subset train --limit-pages 1` with `--subset val`
and choose a fresh output directory such as `outputs/fullpage/dinov2_val`.
Run the two models sequentially initially. Score completed runs with:

```bash
"$LAB_ROOT/.venv/bin/python3" scripts/score_predictions.py \
  outputs/fullpage/dinov2_val/predictions.jsonl \
  outputs/fullpage/dinov3_val/predictions.jsonl \
  outputs/fullpage/dinov2_val/pixel_predictions.jsonl \
  --names dinov2 dinov3 pixel --data-root "$LAB_ROOT/data/d2c_v3" \
  --subset val --out outputs/fullpage/validation_metrics
```

The scorer selects each model's maximum-F1 validation threshold independently.
These are validation-selected results, not independent test estimates. The pixel
predictions should match across backbones because they use identical inputs;
only one copy is scored. DiffSpot is not part of this runner or tuning stage.
