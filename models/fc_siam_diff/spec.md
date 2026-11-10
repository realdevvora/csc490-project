# FC-Siam-diff for Visual Regression Detection: Feature Spec

## Summary
We adapt FC-Siam-diff, a change-detection network built for satellite imagery (Daudt et al., ICIP 2018), to compare a reference screenshot of a webpage with a candidate screenshot. For every pixel, the model says whether it belongs to one of the 13 regression types we care about, and which one. That per-pixel map is then reduced to a regression score, a type, a box and a point in the team's shared prediction format, so it can be scored alongside the other models. Browser rendering differences, such as Firefox or WebKit text anti-aliasing, must not be flagged even though they change many pixels.

The model code and training recipe are reused as-is. Our work is in the data layer: how screenshot pairs become training targets, how pairs of different sizes are aligned, and what the network sees during training.

---

## 1. Intent: how the model solves the problem

**The problem.** Given two screenshots of the same page, decide whether the candidate contains one of these regression types:

| Family | Regression types |
|---|---|
| layout | margin, padding, position, width |
| colour | background_color, text_color |
| typography | font_weight, letter_spacing, font_size, line_height |
| shape | border_width, border_radius |
| appearance | opacity |

**A pixel difference is not a regression.** A browser change alone moves about 97,000 pixels on a typical page, while a real regression often changes fewer than 3,000. So "did anything change?" is the wrong question. The question is "did one of these 13 things change?"

**How FC-Siam-diff works.**
1. The two screenshots go through the same feature extractor (one set of weights shared by both images). It describes each image at four levels of detail, from fine edges to coarse layout.
2. At each level, the network takes the absolute difference between the two images' descriptions. Differences in *features*, not raw pixels, are what let it learn that some pixel differences (anti-aliasing, font hinting) don't matter.
3. A decoder turns those differences back into a full-resolution map that labels every pixel as "not a regression" or one of the 13 types.

**From map to verdict.** The model outputs, for every pixel, a confidence for each of the 14 classes. Separate, swappable rules turn that into one regression score per page, the most likely type, and where the change is (section 5). The yes/no threshold on the score is set by the shared scorer on validation pages, not by us. Description text stays with the later stages in the high-level design.

**Why this model.** It is small (1.35M parameters), fully convolutional (works on any page size), and designed for exactly this "two images, what changed" setup. It takes the number of output classes as a setting, so a 13-type classifier needs no change to the model code. It is the cheapest credible learned baseline to compare against pixel diff, SSIM and the larger models on the roadmap.

---

## 2. Adapting the model to web screenshots

The original code expects satellite image pairs that are the same size, with a hand-drawn yes/no change map. Our data has neither: screenshot sizes often differ, there are no change maps, and we need 13 types rather than yes/no. Five decisions follow.

### 2.1 What the model learns to mark (the training target)
The network learns from a per-pixel answer key. We have to build it.

**Where:**

| Option | What it marks | Problem |
|---|---|---|
| Rectangle around all differing pixels (`diff_bbox`) | A filled box | Coarse; for a layout change it covers most of the page |
| Rectangle around the edited element (`bbox_before`/`after`) | Only the mutated element | Misses content that moved as a result |
| Image-level label only | Nothing per pixel | Drops the model's main training signal; needs extra parts |
| **Exact pixel diff (chosen)** | Every pixel that differs | Covers moved content for layout changes |

**What label:** 14 classes instead of the original yes/no. Class 0 is "not a regression"; classes 1–13 are the regression types. The model is set up with `SiamUnet_diff(3, 14)`.

**Rule.**
- In a regression pair, each pixel where any colour channel differs gets that pair's regression type.
- Everything else is class 0, including every difference in `benign` and `no_change` pairs.

**Why.**
- **The "where" reproduces the dataset's own numbers.** The pixel-diff rule gives exactly the `changed_px` and `diff_bbox` values on all 7,271 pairs. So the target is the dataset's existing definition of change, not a threshold we picked.
- **Benign differences are labelled "not a regression".** That's the core teaching signal: it tells the model that browser differences don't count.
- **13 types, not yes/no, to flag only what's on the list.** A difference that resembles none of the types falls into class 0.
- **It fixes an imbalance in the yes/no setup.** On training pages, padding, width and margin make up 54% of all changed pixels; text_color, font_weight and border_radius together make up 2.5%. A single "changed" class is effectively trained on layout changes. With one class per type and its own loss weight (section 2.4), rare types like text-colour changes are no longer drowned out by padding changes.
- **It gives the `category` output for free**, with no extra model head.

**Trade-offs.**
- For layout regressions that push content down (about 6% of regressions), the target covers everything that moved, not only the edited element. Localization will be broad for those cases.
- Look-alike types such as margin and padding may be confused. Results are reported at both type and family level.
- border_radius is thin (171 training pairs, 0.1% of changed pixels) and will be the least reliable type.

### 2.2 Aligning screenshots of different sizes
1,268 pairs have different sizes, mostly in height. The network needs both inputs the same size.

**Rule (chosen):** pad both images to the larger size with white, anchored at the top-left. This is the alignment the dataset itself used; cropping both to the overlap or padding with black do not reproduce its numbers.

**Why it matters for review.** Size mismatch is strongly tied to the label (benign 46%, regression 14%, no_change 3%). The alignment rule is applied identically to every pair, with no decision based on whether sizes match, so the model can't learn "padded means benign" from how we prepare the data. The white band itself is still visible, which is a known residual risk (section 3.4).

### 2.3 What the network sees during training
Screenshots are up to 1304 × 3278 px; a typical regression covers about 50 × 50 px.

| Option | How it works | Trade-off |
|---|---|---|
| **Squares where the images differ (chosen, default)** | 16 squares of 96 × 96 px per pair, centred on differing pixels | Full detail; training time goes to differences, including browser noise |
| Random squares (baseline) | 16 random 96 × 96 px squares per pair per pass | Full detail; fast; most squares show no change |
| All squares (original code) | Every 96 px square, overlapping by half; about 780 per pair | Full detail, but 49× more work per pass (4.8M squares) |
| Whole page, shrunk | Page resized to 1024 × 512 | Fast and simple, but small changes shrink to a few pixels |

All modes sit behind one setting (`INPUT_MODE`), so we can switch and compare without code changes. Whatever the training mode, evaluation always runs the model on whole pages (shrunk, in whole-page mode), because the network accepts any image size.

**Rule.** All 16 squares per pair are centred on pixels where the two screenshots differ. `no_change` pairs, where nothing differs, fall back to random squares, so the model still sees "identical means not a regression". A setting can mix in random squares if the model misbehaves on unchanged areas; it starts at 0.

**Why.**
- **Random squares waste most of the training.** Only about 1% of training pixels are changed, so random squares mostly land where nothing differs. Loss weighting doesn't fix that: it only makes changed pixels count more once a square happens to contain them.
- **Squares are chosen by the raw difference, not the label.** That gives the model the examples that matter: real changes in regression pairs, and browser noise labelled "not a regression" in benign pairs. Choosing by the regression label would put "something differs" squares only in regression pairs and teach the shortcut "any visible difference is a regression".

**Trade-off.** Pairs with large differences (benign, layout) give their squares more places to land, but a small text edit still gets 16 squares because the count per pair is fixed. The difference map is already computed when a pair loads, so the extra cost is negligible. Random squares stay in the notebook as the baseline.

### 2.4 Training recipe and guardrails
- **Recipe from the original notebook:** weighted per-pixel loss, Adam with weight decay 1e-4, learning rate decaying 5% per pass, batch 32, 50 passes.
- **Loss weights per type, tunable.** The original formula weights two classes: "not a regression" at 0.20 and "changed" at 1.98 on our training pages (`FP_MODIFIER = 10`). We keep the "not a regression" weight and split the "changed" weight across the 13 types, using the `changed_px` counts on training pages only. The combined weight on regression pixels stays the same as in the original formula.
  - `TYPE_WEIGHT_POWER` sets how strongly rare types are boosted. At 0 every type gets 1.98, as in the original. At 1 each type carries the same total weight, which gives padding 0.62 and border_radius 165. The default of 0.5 gives padding 1.26, text_color 6.6 and border_radius 20.
  - `TYPE_WEIGHT_OVERRIDE` sets one type's weight by hand, e.g. `{'border_radius': 20}`.
  - Both are tuned on validation pages using per-type accuracy, and changing them needs retraining.
- **Split by page, from `splits_85_15.json`:** 411 training pages, 73 validation pages. Every variant of a page stays on one side, consistent with ADR 0003.
- **Pixels only.** The network receives image tensors and nothing else. File names (which contain the label), sizes and browser names are used only to build targets. Every image is converted to plain RGB, which removes a leak we found: all Firefox and WebKit screenshots carry a transparency channel that no other screenshot has.
- **Model selection on validation pages**, not on training data as the original notebook did. The best checkpoint is chosen by pixel-level F1.

### 2.5 The pixel diff as a veto at evaluation
The raw pixel diff can stop a flag, never cause one. Behind a `GATE_WITH_DIFF` switch, so results are reported with and without it:
- **No difference, no flag.** This is exact on this dataset: all 484 `no_change` pairs have zero differing pixels, and every regression has at least one.
- **No flag on identical pixels.** The model's output is limited to pixels that differ, since the training target never marks an identical pixel. This removes stray flags, which would otherwise raise the page's score and stretch its box.

Whether a difference is a regression, and which type, is always the model's decision.

---

## 3. Results, cost and risks

### 3.1 What has been run
A smoke test only: all four input modes, 1 pass, 3–6 training pairs and 5 validation pairs, on a laptop CPU, with the 14-class target. It confirms that the pieces connect, targets match the dataset on a random sample, checkpoints save, the metrics compute, and the prediction file has one line per validation pair. Resuming from a checkpoint has not been exercised yet. **It says nothing about model quality.**

### 3.2 What to expect from a small first run
Run first on a small subset (e.g. 20 pages) before the full set.
- **Overfit check:** on a handful of pairs, the training loss should fall close to zero. If it doesn't, the targets and the model disagree.
- **Early passes:** expect the model to flag almost everything (every validation page counted as a regression, including benign) or nothing at all. The smoke runs already showed both. The two numbers to watch are the false-alarm rate on benign pairs and recall on regressions; they should separate within the first several passes.
- **Signs of success:** benign false-alarm rate dropping well below 100% while regression recall holds. A raw pixel diff can't clear that bar: it catches every regression but also flags every benign pair.
- **Likely pattern:** colour, opacity and font-weight changes will be detected before small spacing changes. Margin and padding will be confused with each other longer than other types. Benign pairs will be the last to stop being flagged.

**What gets reported:**
- During training (the notebook): pixel-level precision, recall and F1 on validation pages, which pick the best checkpoint.
- For comparison with other models (the shared scorer, from our prediction file):
  - AUROC for regression vs benign (the main number) and for regression vs everything else.
  - Benign false-alarm rate at 95% recall.
  - Precision, recall and F1, plus false-alarm rates on `benign` and `no_change`, at the scorer's threshold.
  - Type macro-F1.
  - Pointing accuracy and box overlap with `diff_bbox`.
  - Recall by family, change size and reflow.
- Our own extra, logged every pass: type and family accuracy, per-type accuracy, and the share of correctly flagged pixels with the right type. A margin/padding confusion check is not built yet.

### 3.3 Time and resources for the full run
Target machine: RTX 4080 (16 GB), i7-13700, 64 GB RAM.

| Item | Estimate | Basis |
|---|---|---|
| Loading and preparing one pair | About 40 ms per worker | Measured on the laptop |
| Data loading per pass (6,188 pairs, 12 workers) | About 20–30 s | From the above |
| GPU work per pass (16 squares per pair) | Seconds to a minute | Small model, 99k small squares; not yet measured |
| Validation per pass (1,083 whole pages) | About 1 minute | Not yet measured |
| **50 passes, 16 squares per pair** | **Roughly 1–3 hours** | Same for random or diff squares, and for 2 or 14 classes; to be confirmed by a 2–3 pass timing run |
| 50 passes, all squares | Likely days | 49× the squares of the 16-square modes |
| GPU memory | Well under 16 GB | Training squares are tiny; whole-page validation is the peak |
| Disk | About 1 GB data plus small checkpoints | Masks are computed on the fly, not stored |

Data loading, not the GPU, is the expected bottleneck.

### 3.4 Roadblocks to expect
1. **Benign pairs vs layout regressions send conflicting signals.** 47% of benign training pairs have different heights, so content shifts down the page and is labelled "not a regression". A layout regression that shifts content is labelled with its type. The model has to tell a browser shift from a CSS shift by what the shifted content looks like. This is the hardest part of the task and the most likely reason for high false alarms on benign pairs.
2. **Scattered false pixels can inflate the score and stretch the box.** The top-N score limits the effect of single pixels, and the veto (section 2.5) removes flags on identical pixels but not on differing ones. Expect to tune N and the pixel threshold on validation pages.
3. **The veto depends on pixel-identical re-renders.** That holds here, but on DiffSpot or live sites an unchanged page may still differ slightly. Then the veto rarely triggers and the model carries the whole decision.
4. **The white padding band can still reveal a size mismatch**, which correlates with the label. The design doc's audits (size-stratified accuracy, metadata-only baseline) should be run before results are trusted.
5. **Dataset-specific shortcut.** "Different browser means benign" holds for every benign pair here, but not in DiffSpot. Good validation scores that collapse on DiffSpot would point to this.
6. **No internal test set.** Validation pages select the checkpoint, tune N and the pixel threshold, and report results, so DiffSpot is the only unbiased score.
7. **Third-party code license.** The upstream repo has no license. `siamunet_diff.py` is included unmodified with attribution to its source, for coursework use.

## 4. Out of scope for this change
Description text, the size and metadata audits, and DiffSpot evaluation. These follow once a full training run exists. Pixel-level overlap scores are not in the shared scorer yet, so mask quality is reported only in our own notebook.

---

## 5. Output specification
After training, the model is run on every validation page and its output is written in the team's shared prediction format, so it is scored by the same scorer as the other three models. That format and its metrics are defined in the shared scoring spec; this section only says how FC-Siam-diff fills it.

### 5.1 From the model to the file
1. **Model output.** For each pair, a confidence map of 14 × page height × page width: for every pixel, a probability for "not a regression" and each of the 13 types, summing to 1. The network gives log-probabilities, so they are exponentiated first.
2. **Veto (optional, section 2.5).** With `GATE_WITH_DIFF` on, pixels where the two screenshots are identical are set to "not a regression".
3. **Evaluation rules.** Four swappable rules (section 5.3) reduce the map to one line of the file.
4. **File.** One JSON line per pair, written to `fc_siam_diff_val.jsonl` (and `fc_siam_diff_test.jsonl` when there is a test subset).

### 5.2 Fields
| Field | Type | What FC-Siam-diff writes |
|---|---|---|
| `sample_id` | string | The pair's `sample_id` from the metadata |
| `score` | number in [0, 1] | Regression confidence of the page (rule below) |
| `op_pred` | one of the 13 type names | Most likely regression type; written for every pair, but scored only on true regressions |
| `box` | `[x, y, w, h]` | Main changed region, in original screenshot pixels; `null` if no pixel is flagged |
| `point` | `[x, y]` | Single most likely changed location, in original screenshot pixels; `null` if no pixel is flagged |
| `model` | string | `"fc_siam_diff"` plus the run name; ignored by the scorer |
| `op_family`, `op_second` | string | Family of `op_pred` and second-choice type, for our own analysis; ignored by the scorer |

### 5.3 Evaluation rules
A pixel's *regression confidence* is 1 minus its "not a regression" probability. A pixel is *flagged* if its regression confidence is above a pixel threshold.

| Field | Default rule | Alternatives to compare |
|---|---|---|
| `score` | Mean regression confidence of the N most confident pixels | Highest single pixel; confidence-weighted count of flagged pixels |
| `op_pred` | Type with the largest share of flagged pixels | Vote weighted by confidence; type of the most confident region |
| `box` | Box around all flagged pixels | Box around the largest connected region (`largest_component_box` in the shared code, at 1 px per cell) |
| `point` | Most confident flagged pixel | Centre of the topmost flagged region |

Each rule is chosen by a setting, so rules can be swapped and compared without retraining. If no pixel is flagged, `op_pred` falls back to the type with the highest total confidence over the page.

**Why.**
- **The score must be continuous.** The scorer's main numbers (AUROC, regression vs benign; benign false-alarm rate at 95% recall) don't use a threshold, and it picks the yes/no threshold itself from validation predictions. Averaging the top N pixels instead of taking the single highest keeps one stray pixel from deciding the page.
- **One type per page is enough here.** Every regression pair in the dataset has exactly one mutation, so the largest share is a fair rule.
- **Box around all flagged pixels by default.** `diff_bbox` is the box around every differing pixel (section 2.1), so boxing every flagged pixel should overlap it better than boxing only the largest region. This is an expectation, not a measurement; both are compared.
- **Topmost region as a point option, for reflow.** For pairs where content was pushed down, the scorer checks the point against the edited element's new position (`bbox_after`), not the whole moved area. The edit sits above what it pushes, so the topmost region should find it.

### 5.4 Requirements on the output
- **No labels at prediction time.** The training loader uses the pair's label to build its target. The prediction step loads only the two images.
- **Every validation pair, exactly once.** 1,083 lines: 864 regression, 73 `no_change` and 146 `benign`.
- **Coordinates in original pixels.** Padding is anchored at the top-left and pages aren't resized, so map pixels are screenshot pixels as they are. In whole-page training mode, pages are shrunk, so coordinates are scaled back before writing.
- **Settings tuned on validation pages only.** N, the pixel threshold and the rule choices are picked on validation pages. For test or DiffSpot predictions, the scorer takes the yes/no threshold from our validation file (`--threshold-from`).
- **Settings recorded.** The run's `config.json` stores the rule choices, N and the pixel threshold next to the prediction file.

### 5.5 Example lines
Illustrative values only.
```json
{"sample_id": "d2c_000_font_weight", "score": 0.94, "op_pred": "font_weight", "box": [212.0, 388.0, 340.0, 42.0], "point": [301.0, 405.0], "model": "fc_siam_diff", "op_family": "typography", "op_second": "font_size"}
{"sample_id": "d2c_000_firefox", "score": 0.12, "op_pred": "padding", "box": null, "point": null, "model": "fc_siam_diff", "op_family": "layout", "op_second": "margin"}
{"sample_id": "d2c_000_identical", "score": 0.0, "op_pred": "padding", "box": null, "point": null, "model": "fc_siam_diff", "op_family": "layout", "op_second": "margin"}
```
