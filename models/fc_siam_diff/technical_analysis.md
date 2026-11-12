# FC-Siam-diff for Visual Regression Detection: Technical Analysis

How FC-Siam-diff is used to detect and name CSS regressions between two screenshots, how its output is interpreted, how well it works, and where it is and isn't feasible for this problem. Design decisions are in `spec.md`. Results are on the 1,083 validation pairs (864 regression, 146 benign, 73 no-change) of the earlier split, from the best checkpoint of run 2. The team split now excludes 25 pages (ADR 0004), so these numbers will be replaced after retraining on it.

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
| 4. Page score | How likely the page is to contain a regression, 0–1 | mean of the 50 most confident pixels (alternative: mean confidence over differing pixels) |
| 5. Type | Which of the 13 types | type with the largest share of flagged pixels (alternative: confidence-weighted vote) |
| 6. Box | Where the change is | box around all flagged pixels |
| 7. Point | The single most likely changed location | most confident flagged pixel |

If no pixel is flagged, the box and point are empty, and the type comes from the whole page's confidence.

**How it is scored.**
- **Detection:**
  - **AUROC** of the page score, regression vs benign: how often a regression page scores above a benign one, with no threshold needed.
  - **Flag rate:** share of pages with any flagged pixel.
- **Type:** the predicted type compared with the true type, and with the true family.
- **Location:** whether the predicted point lands inside the edited element.

The shared scorer uses the same prediction file to compare all four models.

---

## 4. Metrics

### 4.1 Detection: does it find the regression?
| Measure | Value |
|---|---|
| Regression pages flagged | 84% |
| Benign pages flagged (false alarm) | 99% |
| No-change pages flagged (false alarm) | 3% |
| AUROC regression vs benign, top-50 score (current default) | 0.27 |
| AUROC regression vs benign, mean confidence over differing pixels | **0.69** |
| AUROC regression vs benign, counting differing pixels (no model) | 0.91 |
| Pixel precision / recall | about 0.8 / 0.1 |

The top-50 score saturates on benign pages, which have about 100,000 differing pixels to choose from, so it ranks them *above* regressions. Averaging over the differing pixels uses the same model output and separates the two much better.

### 4.2 Type: does it name the right regression?
| | Ours | Chance or simple baseline |
|---|---|---|
| Exact type (13 options) | **33%** | 8% |
| Family (5 options) | **53%** | 33% (always guess "layout") |

### 4.3 By regression type
- **Flagged:** share of that type's pages with any flagged pixel.
- **AUROC:** that type's pages vs benign pages, using mean confidence over differing pixels.
- **Point on element:** share of pages whose predicted point lands inside the edited element.
- **Colour change:** median change of a differing pixel, 0–255. Browser noise between browsers is **67**.

| Type | Flagged | AUROC vs benign | Point on element | Type right | Colour change |
|---|---|---|---|---|---|
| letter_spacing | 1.00 | 0.90 | 0.80 | 0.74 | 115 |
| font_size | 0.98 | 0.85 | 0.70 | 0.02 | 115 |
| margin | 0.97 | 0.84 | 0.54 | 0.00 | 114 |
| position | 0.99 | 0.84 | 0.96 | 0.24 | 120 |
| border_width | 0.97 | 0.81 | 0.66 | 0.46 | 130 |
| font_weight | 0.90 | 0.78 | 0.76 | 0.35 | 102 |
| padding | 1.00 | 0.75 | 0.50 | 0.00 | 117 |
| width | 0.93 | 0.75 | 0.53 | 0.18 | 125 |
| line_height | 0.90 | 0.68 | 0.56 | 0.10 | 92 |
| background_color | 0.85 | 0.54 | 0.82 | 0.86 | 52 |
| border_radius | 0.76 | 0.39 | 0.76 | 0.00 | 45 |
| text_color | 0.42 | 0.35 | 0.39 | 0.35 | 35 |
| opacity | **0.23** | 0.25 | 0.20 | 0.79 | 46 |

An unflagged page still gets a type from the whole page's confidence, which is why opacity is named correctly far more often than it is flagged.

**Main type confusions:**
| True type | Most often called | Why they look alike |
|---|---|---|
| text_color | opacity | Text gets lighter or darker either way |
| font_size, font_weight, margin, width | letter_spacing | Text gets wider or shifts sideways |
| padding, margin, line_height | position | Content shifts |

---

## 5. Strengths and weaknesses

### Strengths
- **Finds changes that move things.** letter_spacing, font_size, position, margin, border_width and padding are flagged on 97–100% of pages and rank well above browser noise (AUROC 0.75–0.90).
- **Says where.** The output is a map, not just a yes/no, so every verdict comes with a location. For position changes, the predicted point lands on the edited element 96% of the time.
- **Almost never flags an unchanged page.** 3% of no-change pages, likely the few "identical" pairs that actually differ.
- **Interpretation is separate from the model.** Score, type, box and point rules can be changed and compared without retraining. Switching the score rule alone took AUROC from 0.27 to 0.69.

### Weaknesses
These are genuine limits of the model, not tuning issues.
1. **It cannot detect minute changes.**
   - opacity (23% flagged), text_color (42%), border_radius (AUROC 0.39) and background_color (AUROC 0.54) rank below or near browser noise.
   - Their pixel changes (35–52) are fainter than the anti-aliasing noise between browsers (67). The model learned that faint change means harmless.
   - So it flags regressions with large pixel differences but misses subtle ones, which are exactly the ones a person would also miss.
2. **It cannot reject browser noise at page level.**
   - 99% of benign pages have at least one flagged pixel. Browser differences cover about 100,000 pixels, so some always get through, even though the model rejects most of them.
   - Making browser noise more costly in training cut flagged noise pixels from 98% to 42%, but made weakness 1 worse.
3. **A pixel count beats it at detection.** Counting differing pixels separates regression from benign pages with AUROC 0.91, against the model's 0.69. In this dataset, benign pairs change the whole page and regressions change one element. That's a dataset shortcut rather than a skill.
4. **It can't tell look-alike types apart.**
   - margin, padding and font_size are flagged reliably but almost never named correctly; they lose to position and letter_spacing.
   - What separates them is page context: margin pushes neighbours, padding grows the box, font size changes height. The model sees local pixels and doesn't get that context.
5. **It marks only a few pixels of each edit.** Pixel recall is about 10%, so boxes are small.
6. **For layout types it points at moved content, not the edit.** The point lands on the edited element only 50–56% of the time for padding, margin, width and line_height.

---

## 6. Usefulness and feasibility

**Where it is useful.**
- As a **change locator**: given two renders in the same browser, it reliably shows where something geometric changed and roughly what family it is.
- As a **first-pass filter** for layout and typography regressions, where it catches nearly all cases.
- As the **cheapest learned baseline**: small, fast, and it produces a per-pixel map that larger models don't give for free.

**Where it is not feasible.**
- **Telling a regression from a browser difference on its own.** The core requirement, not flagging Firefox or WebKit differences, is not met: benign pages are flagged on 99% of pages, and a plain pixel count ranks better.
- **Subtle colour and opacity regressions.** Their signal sits below browser noise, and a pixel-difference model can't separate them by strength alone.
- **Exact type naming for layout regressions.** Without page-level context, margin vs padding vs position is a coin toss at best.

**Scope.** FC-Siam-diff is feasible as a localisation component and as a detector for geometric regressions in same-browser comparisons. On its own it is not sufficient for the full problem. Two things would extend it:
- page-level reasoning for type, such as whole-page input or a second stage working from page features;
- a way to see the *shape* of a change rather than its strength. Browser noise is scattered along text edges, while a colour regression is a uniform shift over a solid area.
