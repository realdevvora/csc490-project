# FC-Siam-diff for Visual Regression Detection: Technical Analysis

How FC-Siam-diff is used to detect and name CSS regressions between two screenshots, how its output is interpreted, how well it works, and where it is and isn't feasible for this problem. Design decisions are in `spec.md`. Results are from run 3, on the 1,033 validation pairs of the team split (826 regression, 138 benign, 69 no-change; ADR 0004), at its best checkpoint. Headline numbers come from the team's shared scorer (`scripts/score_predictions.py`); per-type numbers come from our own diagnostic. Earlier runs are summarised in section 7.

---

## 1. The model, briefly

FC-Siam-diff (Daudt et al., ICIP 2018) is a Siamese convolutional network built for change detection between two images.

- **Siamese encoder.** The reference and candidate screenshots go through the *same* encoder, with one set of shared weights. That makes the two descriptions directly comparable.
- **Difference, not pixels.** At each level of detail, from fine edges to coarse layout, the network takes the absolute difference between the two screenshots' features.
- **Decoder.** The decoder rebuilds a full-resolution map from those differences, combining them level by level, and labels every pixel.

It was built for satellite images. We use it unchanged and adapt only what goes in and what comes out. It was taught that:
- differing pixels inside the edited element are that regression's type;
- differing pixels in benign pairs (Firefox or WebKit vs Chromium) are "not a regression";
- identical pixels are "not a regression".

---

## 2. What it produces

For each screenshot pair, the model outputs a **14 × H × W** map, where H × W is the page size.

- Each pixel gets 14 probabilities that sum to 1: one for "not a regression" and one for each of the 13 regression types.
- It runs on the whole page at once, at full resolution, so every pixel of the page gets an answer.

| Family | Regression types |
|---|---|
| layout | margin, padding, position, width |
| colour | background_color, text_color |
| typography | font_size, font_weight, letter_spacing, line_height |
| shape | border_width, border_radius |
| appearance | opacity |

The map says *where* the model thinks something changed and *what kind* of change it is, pixel by pixel. It does not by itself say whether the page has a regression. That takes an interpretation step.

---

## 3. Interpreting the output (evaluation method)

Each step turns the map into one line of the team's shared prediction format: score, type, box and point. The rules are swappable settings, so they can be compared without retraining.

| Step | What happens | Setting |
|---|---|---|
| 1. Veto | Pixels identical in both screenshots are forced to "not a regression". The raw pixel difference can stop a flag but never cause one | on |
| 2. Pixel confidence | Regression confidence of a pixel = 1 − P(not a regression) | |
| 3. Flag pixels | A pixel is flagged when its regression confidence is above the threshold | 0.5 |
| 4. Page score | How likely the page is to contain a regression, 0–1 | mean regression confidence over the pixels that differ |
| 5. Type | Which of the 13 types | vote over flagged pixels, weighted by confidence |
| 6. Box | Where the change is | box around all flagged pixels |
| 7. Point | The single most likely changed location | most confident flagged pixel |

If no pixel is flagged, the box and point are empty, and the type comes from the whole page's confidence.

### 3.1 From confidences to a verdict
Each pixel *p* has 14 probabilities: P₀(p) for "not a regression" and P₁(p)…P₁₃(p) for the 13 types. Its regression confidence is r(p) = 1 − P₀(p), the total probability of all 13 types together.

**Is it a regression?**
- **Page score** = the mean of r(p) over the pixels that differ between the two screenshots. A page whose differing pixels mostly look like regressions scores near 1; one whose differences look like browser noise scores lower. Pages with no differing pixel score 0.
- **Yes or no** = page score above a threshold. We don't pick this threshold. The shared scorer chooses it on validation pages (the one that maximises F1), and its main numbers (AUROC, false alarms at 95% recall) don't need a threshold at all.
- Averaging over the differing pixels, rather than taking the most confident ones, stops a page with many differing pixels from scoring high just because some of them are confident. Benign pages have about 100,000 differing pixels. With a mean of the 50 most confident pixels, they outscored regressions (AUROC 0.27 vs 0.69, on run 2).

**Which type?** A confidence-weighted vote over the flagged pixels, those with r(p) > 0.5:
1. For each type *k*, add up Pₖ(p) over all flagged pixels.
2. The type with the largest total is the verdict (`op_pred`); the next largest is the runner-up (`op_second`).
3. If no pixel is flagged, the same sum runs over every pixel of the page.

Every flagged pixel counts towards every type in proportion to its probability. A pixel that is 60% letter_spacing and 30% font_size adds to both, not only its top choice. A confident pixel counts more than a borderline one.

**Where?** The box surrounds all flagged pixels. The point is the pixel with the highest r(p).

**Type rules compared** (run 3, validation pages; each can be selected with `OP_RULE` without retraining):
| Rule | How it decides | Type accuracy | Family accuracy |
|---|---|---|---|
| **Confidence-weighted vote (used)** | Sum of each type's probability over all flagged pixels | **0.354** | **0.558** |
| Pixel share | Each flagged pixel votes once for its most likely type; most votes wins | 0.332 | 0.533 |
| Most confident region | Take the connected group of flagged pixels around the most confident pixel and count votes only inside it, to ignore content pushed around elsewhere | 0.322 | 0.545 |

The three rules differ by about 3 points. How votes are counted barely matters; what limits type naming is the model's per-pixel probabilities (section 5).

**How it is scored.** The team's shared scorer reads the prediction file and reports the same metrics for all four models:
- **Detection without a threshold:**
  - **AUROC, regression vs benign:** how often a regression page scores above a benign one. This is the main comparison number.
  - **Benign false-alarm rate at 95% recall:** the share of benign pages flagged when the threshold is set to catch 95% of regressions.
- **Detection at a threshold:** precision, recall, F1 and false-alarm rates, at the threshold that maximises F1 on validation.
- **Type:** macro-F1 over the 13 types, on true regressions.
- **Location:**
  - **Pointing accuracy:** is the point inside the changed area? That's `diff_bbox`, or for reflowed changes the edited element's new box.
  - **Mean box IoU:** overlap of the box with `diff_bbox`.

Our own diagnostic adds:
- per-type detection and type accuracy;
- a stricter pointing check: is the point on the edited element itself, before or after the edit?
- a no-model baseline that ranks pages by how many pixels differ.

---

## 4. Metrics

### 4.1 Detection: does it find the regression?
| Measure | Value |
|---|---|
| AUROC regression vs benign | **0.725** |
| AUROC regression vs all (benign + no-change) | 0.817 |
| Benign false-alarm rate at 95% recall | 0.949 |
| No-change pages flagged | 0% |
| Benign pages with any flagged pixel | 100% |
| Page score, median: regression / benign / no-change | 0.71 / 0.48 / 0.00 |
| AUROC regression vs benign, counting differing pixels (no model) | 0.923 |

**At the scorer's threshold, the result is the same as flagging everything that differs.**
- The threshold that maximises F1 is 0.082. Every regression page scores above it, and so does every benign page.
- That gives recall 1.00, benign false-alarm rate 1.00, precision 0.857 and F1 0.923.
- 0.857 is simply the share of regressions among pages that differ at all. So these numbers measure the veto, not the model, and should not be reported as model skill.
- The threshold-free numbers, AUROC 0.725 and 95% of benign pages flagged at 95% recall, are the honest summary.

### 4.2 Type: does it name the right regression?
| | Ours | Chance or simple baseline |
|---|---|---|
| Exact type, accuracy (13 options) | **35%** | 8% |
| Exact type, macro-F1 (shared scorer) | 0.29 | |
| Family (5 options) | **56%** | 33% (always guess "layout") |

### 4.3 Location: does it find where?
| Measure | Value |
|---|---|
| Pointing accuracy (shared scorer: point inside `diff_bbox`, or the new element box for reflow) | 0.94 |
| Mean box IoU with `diff_bbox` (shared scorer) | 0.92 |
| Point on the edited element itself (our diagnostic) | 0.65 |
| Share of a regression's differing pixels that are flagged, median | 0.93 |

**These location numbers are largely inherited from the pixel difference.**
- The veto means only differing pixels can be flagged, and the model flags a median 93% of them on regression pages.
- So the box around flagged pixels nearly reproduces `diff_bbox`, the box around all differing pixels.
- The stricter check, whether the point is on the edited element itself, is 0.65.

### 4.4 By regression type
- **Flagged:** share of that type's pages with any flagged pixel.
- **AUROC:** that type's pages vs benign pages.
- **Point on element:** share of pages whose point lands on the edited element itself.
- **Colour change:** median change of a differing pixel, 0–255, over the whole dataset. Browser noise between browsers is **67**.

| Type | Flagged | AUROC vs benign | Point on element | Type right | Colour change |
|---|---|---|---|---|---|
| letter_spacing | 1.00 | 0.94 | 0.70 | 0.86 | 115 |
| position | 1.00 | 0.90 | 0.96 | 0.22 | 120 |
| font_size | 0.98 | 0.89 | 0.72 | 0.14 | 115 |
| margin | 1.00 | 0.89 | 0.62 | 0.00 | 114 |
| width | 0.98 | 0.84 | 0.66 | 0.16 | 125 |
| border_width | 0.99 | 0.84 | 0.81 | 0.59 | 130 |
| padding | 1.00 | 0.83 | 0.44 | 0.09 | 117 |
| font_weight | 0.93 | 0.82 | 0.67 | 0.04 | 102 |
| line_height | 1.00 | 0.77 | 0.45 | 0.36 | 92 |
| background_color | 0.90 | 0.54 | 0.88 | 0.84 | 52 |
| border_radius | 0.80 | 0.53 | 0.76 | 0.00 | 45 |
| text_color | 0.48 | 0.32 | 0.48 | 0.45 | 35 |
| opacity | **0.41** | **0.22** | 0.39 | 0.59 | 46 |
| **All** | 0.88 | 0.73 | 0.65 | 0.35 | |

An unflagged page still gets a type from the whole page's confidence, so a type can be named correctly more often than it is flagged.

**Main type confusions:**
| True type | Most often called | Why they look alike |
|---|---|---|
| font_weight (57 of 69), margin (36 of 69), font_size, padding | letter_spacing | Text gets wider or shifts sideways |
| text_color (36 of 69) | opacity | Text gets lighter or darker either way |
| opacity (25 of 69) | text_color | Same, in the other direction |
| width, line_height, font_size | position | Content shifts |

---

## 5. Strengths and weaknesses

### Strengths
- **Finds changes that move things.** All nine geometric and text-layout types are flagged on 93–100% of pages, and rank well above browser noise (AUROC 0.77–0.94). letter_spacing reaches 0.94.
- **Never flags an unchanged page.** 0% of no-change pages. Every one is pixel-identical, so the veto stops them all.
- **Says where.** Every verdict comes with a box and a point. For position changes, the point lands on the edited element 96% of the time.
- **Interpretation is separate from the model.** Score, type, box and point rules can be changed and compared without retraining. Switching the score rule alone took AUROC from 0.27 to 0.69 on run 2's checkpoint.

### Weaknesses
These are genuine limits of the model, not tuning issues.
1. **It cannot detect minute changes.**
   - opacity (41% flagged, AUROC 0.22), text_color (48%, 0.32), border_radius (0.53) and background_color (0.54) rank below or near browser noise.
   - Their pixel changes (35–52 on a 0–255 scale) are fainter than the anti-aliasing noise between browsers (67). The model learned that faint change means harmless.
   - So it flags regressions with large pixel differences well, but misses subtle ones, which are exactly the ones a person would also miss.
2. **It cannot reject browser noise.**
   - Every benign page has flagged pixels. Catching 95% of regressions means flagging 95% of benign pages.
   - It flags about half of each benign page's differing pixels (median 0.47), against 0.93 on regression pages. That's a real signal, but too weak to separate them.
3. **A pixel count beats it at detection.** Counting differing pixels separates regression from benign pages with AUROC 0.92, against the model's 0.73. In this dataset, benign pairs change the whole page and regressions change one element. That's a dataset shortcut rather than a skill.
4. **It can't tell look-alike types apart.**
   - margin, font_weight, padding and font_size are flagged reliably but almost never named correctly; they lose to letter_spacing and position.
   - What separates them is page context: margin pushes neighbours, padding grows the box, font size changes height. The model sees local pixels and doesn't get that context.
5. **Its location numbers mostly come from the pixel difference.** Pointing 0.94 and box IoU 0.92 are high because only differing pixels can be flagged. Whether the point lands on the edited element itself is 0.65, and only 0.44–0.45 for padding and line_height, where it lands on moved content.

---

## 6. Usefulness and feasibility

**Where it is useful.**
- As a **change locator**: given two renders in the same browser, it shows where something changed and, for geometric changes, roughly what family it is.
- As a **detector for layout and typography regressions**: nearly all are flagged, and they rank well above browser noise.
- As the **cheapest learned baseline**: small, fast, and it produces a per-pixel map that larger models don't give for free.

**Where it is not feasible.**
- **Telling a regression from a browser difference on its own.** The core requirement, not flagging Firefox or WebKit differences, is not met. At 95% recall, 95% of benign pages are flagged, and a plain pixel count ranks better.
- **Subtle colour and opacity regressions.** Their signal sits below browser noise, and a pixel-difference model can't separate them by strength alone.
- **Exact type naming for layout regressions.** Without page-level context, margin vs padding vs position is a coin toss at best.

**Scope.** FC-Siam-diff is feasible as a localisation component and as a detector for geometric regressions in same-browser comparisons. On its own it is not sufficient for the full problem. Two things would extend it:
- page-level reasoning for type, such as whole-page input or a second stage working from page features;
- a way to see the *shape* of a change rather than its strength. Browser noise is scattered along text edges, while a colour regression is a uniform shift over a solid area.

---

## 7. Experiments

All training runs used an RTX 4080 (ThinkStation), 50 passes, Adam with the learning rate decaying 5% per pass, and the best checkpoint by pixel F1 on validation pages.

### 7.1 Training runs
| | Smoke tests | Run 0 | Run 1 | Run 2 | Run 3 |
|---|---|---|---|---|---|
| Purpose | Check the pieces connect | First full run | First full run, after the fix | Fix run 1's failures | Retrain on the team split |
| Split | Old (411/73 pages) | Old | Old | Old | Team split, ADR 0004 (390/69 pages, 25 excluded) |
| Training target | Every differing pixel of a regression | Same | Same | Only differing pixels inside the edited element; moved content gets no answer | Same as run 2 |
| Square size | 96 px | 96 px | 96 px | 192 px | 192 px |
| Type weights | Rare types boosted (power 0.5) | Same | Same | All equal | All equal |
| Browser-noise weight | 0.20 | 0.20 | 0.20 | 2.0 | 2.0 |
| Page score / type rule | top-50 mean / share | Same | Same | Same | Mean confidence over differing pixels / confidence-weighted vote |
| Outcome | All 4 input modes run end to end on a laptop CPU, with 3–6 training pairs | Loss became NaN in the first pass | Plateaued from pass 4; diagnosed at pass 20 | Finished; best at pass 40 | Finished; scored with the shared scorer |

### 7.2 What each run showed
- **Smoke tests.** Targets matched the dataset's own `changed_px` and `diff_bbox`, and the prediction file had one line per validation pair. Nothing about quality.
- **Run 0: NaN loss.**
  - A failed Firefox render (`d2c_324_firefox`) is a single flat colour. Scaling it to unit spread divides by zero.
  - Fix: guard the scaling and drop blank renders from training. A scan of all 7,755 screenshots found one more (`d2c_183_firefox`).
- **Run 1: flagged browser noise, collapsed onto three types.**
  - 98% of benign differing pixels flagged; AUROC regression vs benign 0.16.
  - Type accuracy 0.26, family 0.49. letter_spacing, text_color and position were predicted 2–4× more often than they occur.
  - margin, padding and border_radius were never predicted.
  - Changing the type rule made no difference (0.262–0.267), so the problem was training.
- **Run 2: less noise flagged, types spread out, faint changes lost.**
  - Benign differing pixels flagged dropped from 98% to 42%. Type accuracy rose to 0.33 and family accuracy to 0.53. Most of the default labels broke up.
  - margin, padding, font_size and border_radius were still never predicted.
  - opacity and text_color detection fell (23% and 42% flagged): the higher browser-noise weight taught the model that faint change means harmless.
  - Passes 40–50 were flat, so more passes would not help.
- **Run 3: same model, clean data, new page rules.**
  - AUROC regression vs benign 0.725, type accuracy 0.35, family 0.56: the best of all runs.
  - No-change false alarms went from 2.7% to 0.
  - Faint types improved slightly but stayed weak: opacity 23% → 41% flagged, text_color 42% → 48%.
  - font_weight's type accuracy collapsed (0.35 → 0.04, mostly called letter_spacing), while border_width (0.46 → 0.59) and line_height (0.10 → 0.36) improved. Look-alike types trade places between runs rather than becoming separable.
  - The overall pattern is unchanged, which confirms that the weaknesses come from the model, not the split.

### 7.3 Analyses on run 2 (no retraining)
| Analysis | Question | Finding |
|---|---|---|
| Page score comparison | Does the score rule, not the model, cause the benign false alarms? | AUROC regression vs benign: top-50 mean 0.27, flagged ÷ differing pixels 0.65, **mean confidence over differing pixels 0.69**. Made the default in run 3 |
| Type rule comparison | Which rule names the type best? | Confidence-weighted vote 0.33, pixel share 0.31, most confident region 0.30. Made the default in run 3 |
| No-model baseline | How far does counting differing pixels get? | AUROC 0.91. Benign pairs change the whole page (median 101,564 px), regressions one element (median 2,434 px) |
| Detection by type | Which regressions are found? | 97–100% flagged for letter_spacing, font_size, position, margin, border_width and padding; 23% for opacity, 42% for text_color |
| Colour-change strength | Why are some types missed? | Missed types change colour by 35–52 (0–255), less than browser noise (67); well-detected types by 92–130. Pixel count does not explain it |
| Labelled pixels per type | Are the never-predicted types starved of training signal? | Only border_radius (0.1% of labelled pixels). margin (5.4%) and padding (7.3%) have more than letter_spacing (2.6%), so they lose to look-alikes, not for lack of data |
| Point on the edited element | Does the predicted point find the edit? | 96% for position; 50–56% for padding, margin, width and line_height, where it lands on moved content |

### 7.4 Data audit
- Of the 484 no-change pairs, 25 were not pixel-identical:
  - 10 have content differences, from 240 to 177,358 px;
  - 15 differ in height by 1–30 px.
- Run 2's two no-change false alarms were among them.
- The team's ADR 0004 now excludes those 25 pages with all their pairs, and run 3 uses that split.
