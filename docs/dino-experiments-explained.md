# Understanding the DINO experiments: A, B, and C

**Regression Lens · CSC490 · 14 November 2026**

This guide explains what the experiments do, why we run them, and how to read
their progress and results. It describes the implemented pipeline. Experiment A
has reported validation results; the current B/C runs are still in progress.

## 1. The problem: different does not always mean broken

Imagine comparing two screenshots of the same webpage. A button has moved in one
pair. In another, the browser renders text slightly differently. Both pairs have
different pixels, but our dataset assigns different meanings to those changes.

| Dataset label | Meaning in this project | Example |
| --- | --- | --- |
| **Regression** | An intentional visible CSS/DOM mutation | Changed text colour, spacing, width, or position |
| **Benign** | A separately generated browser/rendering variation | Firefox or WebKit rendering differs from the reference |
| **No-change** | An identical reference/candidate control | The same image content appears on both sides |

An intentional small mutation is still a regression under our label policy.
We do not automatically call it benign because it is subtle.

We want two outputs:

1. **Detection:** Is this pair likely to contain a regression?
2. **Localization:** Where is the regression most likely to be?

A system can succeed at one and fail at the other. It might point at changed
text correctly but also flag harmless browser variation everywhere else.

## 2. What DINO contributes

DINOv2 and DINOv3 are pretrained visual **backbones**: neural networks that turn
an image into numerical descriptions called **features** or **embeddings**.
Their self-supervised learning happened during pretraining, before our project.
We are not running a self-supervised training objective in experiment A.

Rather than describing the entire image with one number, we use **patch
features**: a vector describing each location on a spatial grid.

| Backbone | Input window | Patch size | Spatial grid | Features per patch |
| --- | --- | --- | --- | --- |
| DINOv2-B/14 | 448 x 448 pixels | 14 x 14 pixels | 32 x 32 = 1,024 patches | 768 numbers |
| DINOv3-B/16 | 448 x 448 pixels | 16 x 16 pixels | 28 x 28 = 784 patches | 768 numbers |

We remove the non-spatial CLS token and, for DINOv3, its four register tokens.
Those tokens are not additional screenshot locations.

Think of the patch grid as a map of visual descriptions. Corresponding positions
in the reference and candidate can be compared. However, a pretrained feature
does not inherently know our project's distinction between a regression and
acceptable browser variation.

**Frozen backbone** means its weights do not change during our experiments.
In B and C, only the much smaller downstream **head** learns.

## 3. The experiment matrix

| Experiment | DINOv2 | DINOv3 | Main research question |
| --- | --- | --- | --- |
| **A: Training-free distances** | Frozen features, no trained head | Same | What do the pretrained features reveal on their own? |
| **B: Main learned model** | Frozen backbone + trained spatial head; benign examples included | Same | Can supervision teach regression detection and localization? |
| **C: Benign-data ablation** | Same head, but benign training examples excluded | Same | How much does explicit benign supervision contribute? |

There are **six DINO configurations**. Comparing v2 with v3 is an analysis across
these configurations, not another training condition. A pixel-difference
baseline provides a simple additional reference.

An **ablation** removes one ingredient while keeping the rest controlled. Here,
C removes benign supervision. It does not remove all negative examples.

## 4. Experiment A: compare features without training a head

For every full-page screenshot pair:

```text
Reference page                     Candidate page
      |                                  |
      +---- Matching sliding windows ----+
                         |
                  Frozen DINO
                         |
             Corresponding patch features
                         |
             Cosine and L2 distances
                         |
           Full-page heatmap + pair score
```

**Cosine distance** measures how much feature directions differ. **L2 distance**
measures their straight-line separation in feature space. The pipeline retains
both maps, but the shared detection score uses cosine distance.

For the pair score, we average the largest **5% of valid cosine-map values**.
This is called the **top-5% mean**. The highest response supplies a predicted
location. Large distances indicate feature differences; they are not calibrated
probabilities that a regression exists.

We also compute a pixel baseline from average absolute RGB differences. It uses
no learned features and asks how far a simple image comparison can take us.

**What A establishes:** a baseline before task-specific learning. Although no
weights are trained, the final detection threshold is selected on validation.
Therefore the thresholded result is still validation-selected.

### What A has shown so far

The reported full validation results are:

| Method | Regression/benign AUROC | Benign FPR at 95% recall | F1 | Pointing accuracy |
| --- | --- | --- | --- | --- |
| DINOv2 | 0.082 | 1.000 | 0.923 | 0.539 |
| DINOv3 | 0.206 | 1.000 | 0.923 | 0.585 |
| Pixel difference | 0.097 | 1.000 | 0.923 | 0.953 |

These rounded numbers were supplied from the lab scorer. All three methods
flagged all 826 regressions **and all 138 benign pairs**, while rejecting the
69 no-change pairs. Their high F1 does not mean they reject benign variation.

The result motivates B and C: can learned supervision improve the distinction
between harmful and harmless differences? Improvement is a hypothesis, not a
guarantee.

Pixel difference is a strong localization baseline. For non-reflow examples,
the target box is itself derived from pixel differences, which gives pixel
pointing an expected geometric advantage. This does not demonstrate semantic
regression detection.

## 5. Experiment B: learn from regressions and benign variation

B keeps DINO frozen and trains a **spatial head**, a small convolutional network
that compares paired patch features.

```text
Reference features + candidate features
                    |
        Normalize and compare features
                    |
             Small spatial head
                /          \
     Window confidence    Localization heatmap
```

The input combines three views of the pair:

- **Absolute difference:** how each feature component changes.
- **Elementwise product:** which feature components agree.
- **Cosine similarity:** how closely the normalized vectors align overall.

Each feature vector has 768 components, so the paired input has
`768 + 768 + 1 = 1,537` channels. The head uses 1x1 and 3x3 convolutions, GELU
activations and dropout. One branch predicts a window regression confidence;
another predicts a spatial localization map.

Implementation terminology: A uses cosine **distance**, while the head input
contains cosine **similarity**. Distance is `1 - similarity`; they contain the
same comparison information with opposite direction. This is not another
experimental condition.

Each B training batch contains:

```text
16 regression-positive windows
 8 benign windows
 8 no-change windows
------------------------------
32 windows per weight update
```

This teaches the head that a large visual difference alone is insufficient:
browser differences can be negative examples too. Sigmoid outputs lie between
zero and one, but are not automatically calibrated real-world probabilities.

### How the head learns where to look

The metadata supplies **weak box supervision**. Patches intersecting the target
box are foreground targets; this is less precise than a pixel-perfect mask.

- Normally the target is `diff_bbox`, the changed-region bounding box.
- For **reflow**, where an edit moves surrounding layout, localization uses
  `bbox_after`, the edited element's resulting box.
- Displaced reflow regions outside that target are ignored for localization
  loss rather than taught as unchanged.
- The window detection target depends on intersection with `diff_bbox`.

The training objective adds **binary cross-entropy (BCE)** for detection to a
foreground/background-balanced localization BCE. A **loss** measures prediction
error; gradient-based optimization adjusts the head to reduce it. The optimizer
is AdamW, with learning rate 0.0003 and weight decay 0.01.

## 6. Experiment C: remove benign supervision

C uses the same backbone, head, losses and update budget as B. Its batches are:

```text
16 regression-positive windows
16 no-change windows
------------------------------
32 windows per weight update
```

C still learns binary detection: regressions are positive and no-change windows
are negative. However, it never receives benign examples during optimization.
It may therefore learn that any visible difference is suspicious.

**Both B and C are evaluated on benign examples.** Otherwise we could not test
whether benign training improves rejection of nuisance differences.

The regression sampling stream and total update counts are matched. B does not
receive extra optimizer updates just because its training pool includes benign
pairs. Its fixed training diagnostic sample also matches C's common sample and
excludes benign pairs.

If B rejects benign variation better than C while retaining regression recall,
that supports the value of explicit benign supervision. If B improves rejection
only by missing more regressions, we must report that tradeoff.

## 7. Why training uses windows instead of entire pages

A **window** is a 448 x 448 square crop at native screenshot resolution. We take
the same coordinates from reference and candidate, so corresponding content
stays aligned. A window location therefore produces two image crops.

Each training pair has two frozen window locations:

1. **Uniform window:** a seeded random position on the reference page.
2. **Anchor window:** a position centred around the known mutation target.

Controls use anchor locations from mutations on the same source page. A random
regression window that misses `diff_bbox` is not forced to be a positive;
background-only regression windows are excluded from the balanced sampler.

Resizing a tall page into a small input can erase tiny text, border and spacing
changes. Keeping an entire page at native scale requires many more patch tokens.
Windows preserve local detail and make frozen-feature caching practical.

The tradeoff is less global context. A 448-pixel square cannot show every
relationship across a long page. Good performance on mutation-guided training
crops is therefore not enough: evaluation must cover complete pages.

### How we know the crop is useful

The crop calculation is deterministic and its exact coordinates are saved in a
**manifest**, alongside source IDs and dataset/split hashes. Tests check target
intersection, coordinate translation, padding and image-size changes. Geometric
coverage checks establish box overlap; representative input previews are still
needed to establish that the crop actually shows the relevant visual change.
A large bounding box can include unchanged content, and a large target can be
only partially visible in one crop.

A crop audit found two zero-coverage regression anchors out of 4,739 in the
currently running v1 experiment. Reference-size clipping missed candidate-only
border strips. Those windows were excluded from positive sampling, not
incorrectly labelled positive. A separate v2 correction repairs those anchors
and one matched control. The running v1 experiment is being preserved; versions
must not be mixed. See the technical protocol for the complete audit.

## 8. Reading the run: 11,818, 2,000, and 1,033

These numbers count different things.

### Stage 1: extract 11,818 training windows

The unchanged split contains 390 training source pages and 5,909 training pairs:

| Label | Training pairs | Validation pairs |
| --- | ---: | ---: |
| Regression | 4,739 | 826 |
| Benign | 780 | 138 |
| No-change | 390 | 69 |
| **Total** | **5,909** | **1,033** |

```text
5,909 training pairs x 2 window locations = 11,818 windows
11,818 windows x 2 sides = 23,636 image-crop inputs per backbone
```

These inputs can repeat image content; they are not 23,636 unique webpages.
Extraction runs the frozen backbone and caches its features. It does not train
the head. B/C and all three seeds reuse that backbone's cache.

Float32 caches are estimated at 69.34 GiB for DINOv2 and 53.11 GiB for DINOv3.
The extractor enforces a per-backbone cap and checks free-space headroom. We do
not cache every full-page feature for the entire dataset.

### Stage 2: train to step 2,000

A **step** is one optimizer update using 32 sampled windows:

```text
2,000 updates x 32 windows = 64,000 window presentations per training run
```

Sampling uses replacement, so windows can appear more than once. This is not
64,000 unique examples and not 2,000 **epochs**. An epoch conventionally means
one pass through a dataset; the fixed-update sampler does not follow that loop.

The log may look like this; the values below are illustrative:

```text
{'step': 250,
 'loss': 0.84,
 'detection': 0.49,
 'localization': 0.35,
 'gradient_norm': 0.22,
 'fixed_training_loss': 0.81,
 'fixed_training_auroc': 0.91}
```

| Field | Meaning | What it does not prove |
| --- | --- | --- |
| `step` | Completed head-weight updates | Number of epochs or unique samples |
| `loss` | Detection loss plus localization loss | Final validation quality |
| `detection` | Window classification error term | Full-page recall or precision |
| `localization` | Error against weak spatial targets | Pixel-perfect segmentation quality |
| `gradient_norm` | Magnitude of the update-driving gradients | A quality score where bigger is better |
| `fixed_training_loss` | Loss on the same small common training sample | Independent generalization |
| `fixed_training_auroc` | Window-level ranking on that sample | Regression/benign validation AUROC |

Logs normally appear every 25 steps, with the first step and saved checkpoints
also logged. The head saves **checkpoints** at 500, 1,000 and 2,000 updates.
A checkpoint is a saved set of weights plus its run configuration.

### Why step 2,000 appears repeatedly

Each trained condition uses seeds 490, 491 and 492. A **random seed** controls
initialization, sampling and dropout randomness so runs are reproducible.

```text
2 backbones x 2 trained conditions x 3 seeds = 12 training runs
12 runs x 3 saved checkpoints = 36 saved heads
```

Seeds test training variability. We do not choose whichever seed looks best and
present it as typical performance.

### Stage 3: evaluate 1,033 validation pairs

These are full-page pair comparisons from 69 validation pages, not 1,033 isolated
images. Many pairs reuse the same reference screenshot.

```text
For each backbone:
  Extract training features once
                 |
  Train B and C, three seeds each
                 |
  Evaluate 18 saved heads on 1,033 validation pairs
                 |
  Select checkpoints and generate reports
```

The 18 heads are two conditions x three seeds x three checkpoints. At validation,
each full-page window's backbone features are extracted once and shared across
those heads, rather than rerunning DINO separately for each checkpoint.

## 9. Full-page inference: no answers supplied to the model

At validation, the runner ignores mutation locations when selecting windows.
It slides 448 x 448 windows across the page with a **stride** of 224 pixels.
Stride is the distance between neighbouring window origins, so these windows
overlap by half their width/height.

```text
Window 1: x =   0 to 448
Window 2: x = 224 to 672
Window 3: x = 448 to 896
... repeat across the width and down the page
```

Overlapping views help reduce dependence on where a change falls relative to a
window boundary. White padding handles edges, and regions outside both real
images are excluded from scoring. Both reference and candidate image extents
are considered so size changes remain observable.

For B/C, localization maps are averaged over overlapping windows and mapped back
to original screenshot coordinates. The strongest localization response supplies
the predicted point. The highest window regression confidence becomes the pair
score. A instead uses its full-page top-5% mean distance score.

The common 2-pixel output grid is a coordinate/aggregation convenience. It does
not give the backbone true 2-pixel feature resolution.

**Inference** means making predictions without updating weights. Annotations are
joined afterward for evaluation; they do not choose the validation windows.

## 10. Reading the metrics without being misled

| Metric | Plain-language question | Interpretation |
| --- | --- | --- |
| **Regression/benign AUROC** | Does a random regression usually score above a random benign pair? | 1 is ideal, 0.5 is chance-level ranking, below 0.5 indicates the opposite ordering; ties count as half |
| **Benign FPR at 95% recall** | If we catch at least 95% of regressions, how many benign pairs do we flag? | Lower is better; 1 means all benign pairs are flagged |
| **Precision** | Of the pairs flagged, how many are regressions? | Higher is better, but depends on the class mix |
| **Recall** | Of the actual regressions, how many did we catch? | Higher is better; flagging everything achieves high recall |
| **F1** | How do precision and recall balance? | Useful at a threshold, but sensitive to class proportions |
| **Benign/no-change FPR** | How often is each negative class falsely flagged? | Report separately; identity controls are much easier |
| **Pointing accuracy** | Is the predicted point inside the known target region? | Measures localization, independently of the detection threshold |

A **threshold** converts a continuous score into a yes/no decision. The scorer
selects a maximum-F1 threshold on validation for each checkpoint. Different
models can need different thresholds. The terminal rounds numbers to three
decimal places; `0.000` is not proof that a threshold is exactly zero.

### Why an F1 of 0.923 can accompany poor detection

Experiment A's thresholded results have:

```text
True positives:  826   regressions correctly flagged
False positives: 138   benign pairs incorrectly flagged
True negatives:   69   no-change pairs correctly rejected
False negatives:   0   regressions missed

Precision = 826 / (826 + 138) = 0.857
Recall    = 826 / 826         = 1.000
F1        = 2 x 826 / (2 x 826 + 138) = 0.923
```

Those numbers look strong until we ask about benign variation: every benign
pair was flagged. This is why regression/benign AUROC and benign FPR@95R are our
main comparison metrics. The label proportions in a real deployment could also
be very different from our regression-heavy validation set.

The current models produce points, not boxes or operator predictions. Missing
IoU/type entries are therefore unavailable metrics, not zero-valued failures.

## 11. Fair comparison and fitting checks

All derivatives of one source webpage stay in the same committed partition.
The 390/69 split is unchanged, with existing exclusions. We do not randomly
split overlapping crops into training and validation; that would risk **data
leakage**, where nearly the same page content appears on both sides.

We hold the downstream recipe constant across backbones: image inputs, sampling
rules, head capacity, losses, optimizer, number of updates and scoring rules.
Their patch grids and pretraining histories still differ. Our conclusion is
about available pretrained representations under a controlled setup, not a
pure architecture comparison.

**Overfitting** is fitting training-specific patterns that do not transfer well.
**Underfitting** is failing to learn enough of the relevant training signal.
Neither can be ruled out by a single loss number or guaranteed absent by design.

We inspect:

- Training loss components and gradients for optimization problems.
- Fixed training diagnostics for whether the head can learn its inputs.
- Full-page validation AUROC and benign rejection at the saved checkpoints.
- Localization, operator/magnitude/change-size breakdowns and review figures.
- Three-seed means and sample standard deviations, rather than one lucky run.

The fixed training AUROC uses local windows and excludes benign pairs. The main
validation AUROC uses full pages and compares regressions with benign pairs.
Their numerical difference is not a clean, like-for-like generalization gap.

A small synthetic overfit test checks whether gradients and optimization work.
Passing it is an engineering check, not evidence of useful dataset performance.

## 12. Checkpoint selection and the final comparison

For each backbone and B/C condition, we select one common checkpoint step across
its three seeds:

1. Highest mean validation regression/benign AUROC.
2. If tied, lower mean benign FPR at 95% recall.
3. If still tied, the earlier checkpoint.

All seed results are retained. Standard deviation measures variability from
training randomness; it is not a confidence interval for performance on unseen
webpages. Multiple pairs from one page are also related observations.

The final report compares A, B and C for both backbones, plus pixel difference.
The most useful conclusions answer specific questions:

- **B versus A:** Does task supervision improve detection/localization?
- **B versus C:** Does benign supervision improve nuisance rejection under a
  matched update budget?
- **DINOv2 versus DINOv3:** Which representation works better under the shared
  downstream recipe, and with what variability and resource cost?
- **DINO versus pixels:** What do learned features add beyond direct differences?

The same 69 validation pages select checkpoints and thresholds. Therefore these
are **validation-selected results**, not independent internal-test estimates.
We are not introducing a new split or evaluating DiffSpot in this round.

The strongest outcome is a reproducible explanation of what works, what fails,
and what supervision changes—not simply the largest number in a results table.

## Technical references

- [Full-page baseline policy and results](results/dino/fullpage-run.md)
- [Frozen-head recipe, crop audit and launch instructions](results/dino/head-runs.md)
- [Shared prediction format and metric definitions](prediction-format.md)
- [Project status](project-status.md)

The current running v1 results use `outputs/dino_heads/`. The prepared corrected
v2 launcher uses `outputs/dino_heads_v2/`. Preserve their manifests and reports
separately; do not replace a running experiment's inputs or combine versions.
