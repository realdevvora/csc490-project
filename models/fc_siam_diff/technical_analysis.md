# FC-Siam-diff for Visual Regression Detection: Technical Analysis

What the trained model does, the settings it runs with, how well it works on each regression type, and where it fails. The design decisions behind it are in `spec.md`. All results are on the 1,083 validation pairs (73 pages), from run 2 at its best checkpoint (pass 40 of 50) unless stated otherwise.

**In one line:** the model reliably flags regressions that *move* things (spacing, size, position, borders), but it cannot tell *faint* changes (colour, opacity, rounded corners) apart from browser rendering noise, and it cannot tell look-alike layout types apart.

---

## 1. The model

### 1.1 Architecture
FC-Siam-diff (Daudt et al., ICIP 2018), used unmodified from the upstream code (`siamunet_diff.py`), set up as `SiamUnet_diff(3, 14)`.

| Part | What it does |
|---|---|
| Input | Two RGB screenshots of the same page, reference and candidate, each scaled to mean 0 and spread 1 |
| Encoder | One feature extractor with shared weights, applied to both screenshots. Four levels of detail with 16, 32, 64 and 128 channels, from fine edges to coarse layout |
| Comparison | At each level, the absolute difference between the two screenshots' features |
| Decoder | Upsamples back to full resolution, combining those differences level by level |
| Output | For every pixel, 14 probabilities: "not a regression" plus the 13 regression types |
| Size | 1,351,886 trainable parameters. Fully convolutional, so it runs on a whole page of any size in one pass |

### 1.2 How a prediction is made
1. **Align.** If the two screenshots differ in size, both are padded with white to the larger size, anchored at the top-left.
2. **Run the network** on the whole page. Output: 14 × page height × page width.
3. **Veto.** Pixels that are identical in both screenshots are forced to "not a regression". The pixel diff can stop a flag, never cause one.
4. **Per-pixel confidence.** A pixel's regression confidence is 1 − P(not a regression). It is *flagged* above 0.5.
5. **Page verdict.** Rules reduce the map to one line of the team's prediction format: a score, a type, a box and a point.

### 1.3 How it was trained
| Aspect | Choice |
|---|---|
| Training target | In a regression pair, differing pixels inside the edited element's box (before or after) get the regression type. Differing pixels outside it (content that only moved) get no answer and add nothing to the loss. Differing pixels in benign or no_change pairs are "not a regression". Identical pixels are "not a regression" |
| What the network sees | 16 squares of 192 × 192 px per pair, centred on differing pixels that have an answer; random squares when nothing differs. Random horizontal flips |
| Loss | Weighted per-pixel loss from the original notebook, plus a separate weight for browser-noise pixels |
| Optimiser | Adam, learning rate decaying 5% per pass, weight decay 1e-4, batch 32 squares, 50 passes |
| Checkpoint | Best pixel F1 on validation pages (pass 40) |
| Hardware and time | RTX 4080, about 180–250 s per pass, about 2.5–3.5 hours in total |

---

## 2. Settings and thresholds

| Setting | Value | Meaning |
|---|---|---|
| Square size | 192 px | Size of each training square (run 1 used 96) |
| Squares per pair | 16 | Per pair, per pass |
| Weight on "not a regression" pixels | 0.20 | From the original formula (`FP_MODIFIER = 10`) |
| Weight on each regression type | 1.98 | All 13 types equal (`TYPE_WEIGHT_POWER = 0`) |
| Weight on browser-noise pixels | 2.0 | Differing pixels in benign and no_change pairs (`BENIGN_DIFF_WEIGHT`) |
| Moved content | No answer | `LABEL_MOVED_CONTENT = False` |
| Pixel threshold | 0.5 | Regression confidence above which a pixel is flagged |
| Page score | Mean of the 50 most confident pixels | `SCORE_RULE = 'top_n'`, `TOP_N = 50`. Section 4.1 shows this rule is the wrong choice |
| Page type | Type with the largest share of flagged pixels | `OP_RULE = 'share'`. `weighted` (confidence-weighted vote) is slightly better, section 5 |
| Box | Box around all flagged pixels | `BOX_RULE = 'all'` |
| Point | Most confident flagged pixel | `POINT_RULE = 'peak'` |
| Page flagged | At least one flagged pixel | Used by the notebook's image metrics. The shared scorer sets its own threshold on the score |

---

## 3. Data

| | Train | Validation |
|---|---|---|
| Pages | 411 | 73 |
| Regression pairs | 4,955 | 864 |
| Benign pairs (Firefox or WebKit vs Chromium) | 821 | 146 |
| No-change pairs (Chromium re-render) | 411 | 73 |

One blank Firefox render (`d2c_324_firefox`) is dropped from training.

**Observations about the data that affect the results:**
- **Benign and regression pairs differ hugely in size of change.** A median of 101,564 differing pixels for benign pairs vs 2,434 for regressions.
- **Regression types label very different amounts of training signal.** border_radius is 0.1% of all labelled training pixels and background_color 30.4%.
- **Not all "identical" pairs are identical.** 10 of 484 no_change pairs have content differences, 8 of them visible. 15 more differ only in page height by 1–30 px.

---

## 4. Detection: does it flag the regression?

### 4.1 Overall
| Measure | Value | Comment |
|---|---|---|
| Regression pages flagged | 84% | |
| Benign pages flagged (false alarm) | 99% | |
| No-change pages flagged (false alarm) | 3% | Likely the non-identical "identical" pairs |
| AUROC regression vs benign, current score (top-50 mean) | 0.27 | Benign pages score *higher* than regressions |
| AUROC regression vs benign, mean confidence on differing pixels | **0.69** | Same model, better page score; not yet the default |
| AUROC regression vs benign, flagged ÷ differing pixels | 0.65 | |
| AUROC regression vs benign, pixel count only (no model) | **0.91** | Fewer differing pixels → regression |
| Overall page accuracy | 0.70–0.77 | Below the 79.8% from calling every page a regression |
| Pixel precision / recall | about 0.8 / 0.1 | Passes 38–50. Flagged pixels are usually right, but only about 10% of edited pixels are flagged |
| Right type among correctly flagged pixels | about 0.8 | |

### 4.2 By regression type
"Flagged" is the share of that type's pages with any flagged pixel. AUROC is that type's pages vs benign pages, using mean confidence on differing pixels. "Point on element" is the share of pages whose predicted point lands inside the edited element. "Colour change" is the median colour change of a differing pixel, 0–255. Benign browser noise is 67.

| Type | n | Flagged | AUROC vs benign | Point on element | Type right | Colour change | Differing px (median) |
|---|---|---|---|---|---|---|---|
| letter_spacing | 70 | 1.00 | 0.90 | 0.80 | 0.74 | 115 | 1,791 |
| font_size | 66 | 0.98 | 0.85 | 0.70 | 0.02 | 115 | 2,411 |
| margin | 72 | 0.97 | 0.84 | 0.54 | 0.00 | 114 | 3,653 |
| position | 72 | 0.99 | 0.84 | 0.96 | 0.24 | 120 | 2,354 |
| border_width | 70 | 0.97 | 0.81 | 0.66 | 0.46 | 130 | 3,615 |
| font_weight | 71 | 0.90 | 0.78 | 0.76 | 0.35 | 102 | 1,470 |
| padding | 72 | 1.00 | 0.75 | 0.50 | 0.00 | 117 | 12,811 |
| width | 68 | 0.93 | 0.75 | 0.53 | 0.18 | 125 | 4,177 |
| line_height | 61 | 0.90 | 0.68 | 0.56 | 0.10 | 92 | 1,416 |
| background_color | 72 | 0.85 | 0.54 | 0.82 | 0.86 | 52 | 4,282 |
| border_radius | 29 | 0.76 | 0.39 | 0.76 | 0.00 | 45 | 240 |
| text_color | 71 | 0.42 | 0.35 | 0.39 | 0.35 | 35 | 795 |
| opacity | 70 | **0.23** | 0.25 | 0.20 | 0.79 | 46 | 2,066 |
| **All** | 864 | 0.84 | 0.69 | 0.62 | 0.33 | | 2,434 |

"Type right" uses the `weighted` rule and counts every regression page, flagged or not. An unflagged page still gets a type from the whole page's confidence. That is why opacity is named correctly 79% of the time but flagged only 23% of the time.

**The dividing line is how faint the change is, not how many pixels change.** Every type whose colour change is fainter than browser noise (text_color 35, border_radius 45, opacity 46, background_color 52, vs noise 67) is detected badly. Every type that moves edges (92–130) is detected well. Pixel count does not explain it: opacity changes 2,066 pixels and is flagged 23% of the time, while font_weight changes 1,470 and is flagged 90% of the time.

---

## 5. Type: does it name the right regression?

| Rule | Type accuracy (13 options) | Family accuracy (5 options) |
|---|---|---|
| `share` (default) | 0.31 | 0.52 |
| `weighted` | **0.33** | **0.53** |
| `region` | 0.30 | 0.51 |
| Chance / always guess the largest | 0.08 | 0.33 |

**Main confusions (`weighted` rule):**
| True type | Most often called | Why they look alike in pixels |
|---|---|---|
| text_color | opacity (42 of 71) | Text becomes lighter or darker either way |
| font_size, font_weight, margin, width | letter_spacing | Text gets wider or shifts sideways |
| padding, margin, line_height | position | Content shifts |
| border_radius | spread across background_color, font_weight, opacity | Too few pixels to learn from |

---

## 6. Weaknesses

These are genuine weaknesses of the model as built, not tuning issues.

1. **It cannot detect minute changes.**
   - **The symptom.** opacity (23% flagged), text_color (42%), border_radius (AUROC 0.39) and background_color (AUROC 0.54) rank *below* browser noise or close to it.
   - **The cause.** Their pixel changes are fainter than the anti-aliasing noise between browsers. The model was trained that browser noise is harmless, so it learned "faint change = harmless". Raising the browser-noise weight cut false alarms but made this worse.
   - **What it means.** The model flags regressions with large, sharp pixel differences well, but misses subtle colour regressions, which are often the ones a person would also miss.
2. **It cannot reject browser noise at the page level.**
   - 99% of benign pages have at least one flagged pixel. The model now flags 42% of benign differing pixels (run 1: 98%), but browser differences cover about 100,000 pixels, so some always get through.
   - With the top-50 score, benign pages outscore regressions (AUROC 0.27). A score averaged over the differing pixels raises this to 0.69.
3. **It is beaten by a no-model baseline.** Counting differing pixels separates regression from benign pages with AUROC 0.91, better than the model's 0.69. This is a property of the dataset: benign pairs change the whole page, regressions one element. It applies to every model on this dataset, so the pixel-count baseline should be reported next to each.
4. **Look-alike types are never predicted.**
   - margin, padding and font_size have plenty of training pixels, more than letter_spacing. But their pixels look like position or letter_spacing, so they lose every tie.
   - What tells them apart is page context: margin pushes neighbours, padding grows the element's box, font_size changes text height. That context is cropped out of a 192 px square or carries no answer.
   - The same happened in run 1, where moved content was labelled, so it is not caused by the run 2 target.
5. **It marks very little of each edit.** Pixel recall is about 10%. The model flags a few confident pixels per edit rather than the whole change, so the boxes are small.
6. **It points at moved content, not the edit, for layout types.** The predicted point lands on the edited element 96% of the time for position, but only 50–56% for padding, margin, width and line_height.
7. **border_radius is starved.** 200 pairs (others have about 470), about 280 pixels each, 0.1% of the training signal.
8. **Page accuracy is below a trivial baseline.** 0.70–0.77, vs 0.80 for calling everything a regression. Overall page accuracy and F1 should not be reported as headline numbers.

---

## 7. What would address them

| Weakness | Change | Cost |
|---|---|---|
| Page score (2) | Use mean confidence on differing pixels as the score, and `weighted` for the type | Rule change only, no retraining |
| Faint changes (1) | Give the model enough context to see the *shape* of a change: browser noise is scattered along text edges, a colour regression is a uniform shift over a solid area. Bigger squares or whole-page training | Retraining |
| Faint changes (1) | Boost opacity, text_color and background_color with `TYPE_WEIGHT_OVERRIDE`, or lower the browser-noise weight | Retraining; raises benign false alarms again |
| Look-alike types (4) | Whole-page input, or a small second stage that decides the type from page features (did content below shift, did the box grow) | New work beyond the current design |
| border_radius (7) | Higher weight via `TYPE_WEIGHT_OVERRIDE` | Retraining; limited by data |
| Shortcut (3) | Report the pixel-count baseline for all four models; regression pairs rendered in a different browser would remove it | Dataset change, team decision |

---

## 8. Run 1 vs run 2

| | Run 1 | Run 2 |
|---|---|---|
| Training target | Every differing pixel of a regression gets its type | Only differing pixels inside the edited element; moved content gets no answer |
| Square size | 96 px | 192 px |
| Type weights | Rare types boosted (power 0.5) | All equal |
| Browser-noise weight | 0.20 | 2.0 |
| Benign differing pixels flagged | 98% | 42% |
| AUROC regression vs benign (top-50 score) | 0.16 | 0.27 |
| Type / family accuracy (`weighted`) | 0.27 / 0.49 | 0.33 / 0.53 |
| Types never predicted | margin, padding, border_radius | margin, padding, font_size, border_radius |
| Most over-predicted type (`share` rule) | letter_spacing (290 calls for 70 pages) | letter_spacing (242 for 70) |

---

## 9. Key numbers for a poster

- **1.35M** parameters; one pass over a whole page; **14** classes per pixel.
- **97–100%** of letter_spacing, font_size, position, margin, border_width and padding regressions flagged.
- **23%** of opacity and **42%** of text_color regressions flagged: faint changes fall below browser noise (colour change 35–46 vs 67).
- AUROC regression vs benign **0.69** with a differing-pixel score, vs **0.91** for counting pixels alone.
- Type accuracy **33%** (chance 8%), family accuracy **53%** (baseline 33%).
- Browser-noise pixels flagged: **98% → 42%** from run 1 to run 2.
