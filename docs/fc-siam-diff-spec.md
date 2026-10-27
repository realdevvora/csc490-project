# FC-Siam-diff for Visual Regression Detection: Feature Spec

## Summary
We adapt FC-Siam-diff, a change-detection network built for satellite imagery (Daudt et al., ICIP 2018), to compare a reference screenshot of a webpage with a candidate screenshot. For every pixel, the model says whether it belongs to one of the 13 regression types we care about, and which one. A page is flagged if any pixel is assigned a regression type. Browser rendering differences, such as Firefox or WebKit text anti-aliasing, must not be flagged even though they change many pixels.

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

**From map to verdict.** A page is flagged if any pixel is labelled with a regression type. Its category is the most common type among those pixels, and the family follows from the type. The map also says *where* the regression is, which feeds later localization work. Description text stays with the later stages in the high-level design.

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

**What label (not built yet):** 14 classes instead of the original yes/no. Class 0 is "not a regression"; classes 1–13 are the regression types. The model is set up with `SiamUnet_diff(3, 14)`. The notebook currently trains the 2-class version.

**Rule.**
- In a regression pair, each pixel where any colour channel differs gets that pair's regression type.
- Everything else is class 0, including every difference in `benign` and `no_change` pairs.

**Why.**
- **The "where" reproduces the dataset's own numbers.** The pixel-diff rule gives exactly the `changed_px` and `diff_bbox` values on all 7,271 pairs. So the target is the dataset's existing definition of change, not a threshold we picked.
- **Benign differences are labelled "not a regression".** That's the core teaching signal: it tells the model that browser differences don't count.
- **13 types, not yes/no, to flag only what's on the list.** A difference that resembles none of the types falls into class 0.
- **It fixes an imbalance in the yes/no setup.** On training pages, padding, width and margin make up 54% of all changed pixels; text_color, font_weight and border_radius together make up 2.5%. A single "changed" class is effectively trained on layout changes. With one class per type and its own loss weight (section 2.4), a text-colour change counts as much as a padding change.
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
| **Squares where the images differ (chosen; not built yet)** | 16 squares of 96 × 96 px per pair, centred on differing pixels | Full detail; training time goes to differences, including browser noise |
| Random squares (current notebook default) | 16 random 96 × 96 px squares per pair per pass | Full detail; fast; most squares show no change |
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
- **Loss weights per type (not built yet).** The original formula weights two classes. We extend it to one weight per type, computed from the same `changed_px` counts on training pages only. That's about 5 lines of new code. The current 2-class version weights a changed pixel about 10× as much as an unchanged one (`FP_MODIFIER = 10`).
- **Split by page, from `splits_85_15.json`:** 411 training pages, 73 validation pages. Every variant of a page stays on one side, consistent with ADR 0003.
- **Pixels only.** The network receives image tensors and nothing else. File names (which contain the label), sizes and browser names are used only to build targets. Every image is converted to plain RGB, which removes a leak we found: all Firefox and WebKit screenshots carry a transparency channel that no other screenshot has.
- **Model selection on validation pages**, not on training data as the original notebook did. The best checkpoint is chosen by pixel-level F1.

### 2.5 The pixel diff as a veto at evaluation (not built yet)
The raw pixel diff can stop a flag, never cause one. Behind a `GATE_WITH_DIFF` switch, so results are reported with and without it:
- **No difference, no flag.** This is exact on this dataset: all 484 `no_change` pairs have zero differing pixels, and every regression has at least one.
- **No flag on identical pixels.** The model's output is limited to pixels that differ, since the training target never marks an identical pixel. This removes stray flags, which matter because one flagged pixel flags the page.

Whether a difference is a regression, and which type, is always the model's decision.

---

## 3. Results, cost and risks

### 3.1 What has been run
A smoke test only: the three built input modes, 2 passes, 4–6 pairs, on a laptop CPU, with the 2-class target. It confirms that the pieces connect, masks match the dataset on a random sample, checkpoints save, and the metrics compute. Resuming from a checkpoint has not been exercised yet. **It says nothing about model quality.**

### 3.2 What to expect from a small first run
Run first on a small subset (e.g. 20 pages) before the full set.
- **Overfit check:** on a handful of pairs, the training loss should fall close to zero. If it doesn't, the targets and the model disagree.
- **Early passes:** expect the model to flag almost everything (every validation page counted as a regression, including benign) or nothing at all. The smoke runs already showed both. The two numbers to watch are the false-alarm rate on benign pairs and recall on regressions; they should separate within the first several passes.
- **Signs of success:** benign false-alarm rate dropping well below 100% while regression recall holds. A raw pixel diff can't clear that bar: it catches every regression but also flags every benign pair.
- **Likely pattern:** colour, opacity and font-weight changes will be detected before small spacing changes. Margin and padding will be confused with each other longer than other types. Benign pairs will be the last to stop being flagged.

**What gets reported:**
- Flag/no-flag accuracy, precision and recall.
- False-alarm rates on `benign` and `no_change` pairs, separately.
- Pixel-level precision, recall and F1.
- Per-type and per-family accuracy (not built yet).

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
2. **"Any pixel flagged" is a fragile verdict.** One stray predicted pixel flags a whole page. The veto (section 2.5) removes stray flags on identical pixels but not on differing ones, so expect to tune a minimum flagged area on validation pages.
3. **The veto depends on pixel-identical re-renders.** That holds here, but on DiffSpot or live sites an unchanged page may still differ slightly. Then the veto rarely triggers and the model carries the whole decision.
4. **The white padding band can still reveal a size mismatch**, which correlates with the label. The design doc's audits (size-stratified accuracy, metadata-only baseline) should be run before results are trusted.
5. **Dataset-specific shortcut.** "Different browser means benign" holds for every benign pair here, but not in DiffSpot. Good validation scores that collapse on DiffSpot would point to this.
6. **No internal test set.** Validation pages both select the checkpoint and report results, so DiffSpot is the only unbiased score.
7. **Third-party code license.** The upstream repo has no license, so `siamunet_diff.py` is not committed. The team needs to decide whether to vendor it with attribution.

## 4. Out of scope for this change
Description text, the size and metadata audits, per-magnitude breakdowns, and DiffSpot evaluation. These follow once a full training run exists.
