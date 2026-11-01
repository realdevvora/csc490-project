# Regression Lens — Real-World QA Application & Multi-Defect Plan

## Core idea

Regression Lens can be used as an **AI-assisted visual QA system**.

The current dataset is simplified so that each screenshot pair contains a **single controlled mutation**. This makes training labels clean and easy to verify.

That does **not** mean the final system must only detect one defect at a time.

A realistic use case is:

```text
Developer changes HTML/CSS
        │
        ▼
Playwright renders updated webpage
        │
        ├──────── Reference screenshot
        │
        └──────── Actual screenshot
                      │
                      ▼
               Regression Lens
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
      Defect #1    Defect #2    Defect #3
      button       heading       image
      shifted      wrong font    stretched
          │           │            │
          └───────────┴────────────┘
                      ▼
              QA regression report
```

The goal is to help a QA engineer quickly identify suspicious visual changes rather than manually comparing the entire page.

---

# Why single-mutation training still makes sense

Single-mutation examples give us **clean ground truth**.

For example:

```text
Page A
+ only font-size change
```

We know exactly:

- what changed;
- where it changed;
- what mutation type produced it;
- what region should be highlighted.

This makes single-mutation samples useful for learning the basic visual patterns of regressions.

Examples:

```text
font-size change
colour change
spacing change
opacity change
border-radius change
alignment change
asset distortion
missing component
```

These can be viewed as the **atomic building blocks** of more realistic pages containing multiple simultaneous defects.

---

# Real-world multi-defect scenario

A production webpage may contain several regressions at once.

Example:

```text
Original webpage
      │
      ├── button radius changed
      ├── heading font size changed
      └── image width changed
```

A realistic system should ideally detect and report each issue separately.

Example output:

```text
3 possible visual regressions detected

1. Typography
   Header font size differs from reference.
   Confidence: 0.93

2. Layout
   Primary CTA is shifted to the right.
   Confidence: 0.89

3. Media
   Product image aspect ratio differs from reference.
   Confidence: 0.87
```

---

# Recommended research progression

## Experiment 1 — Single regression

```text
Train:
single mutation

Test:
single mutation
```

Purpose:

- establish baseline performance;
- verify the model can detect and localize controlled defects;
- compare model architectures fairly.

---

## Experiment 2 — Multiple regressions

```text
Train:
single mutation

Test:
2–4 simultaneous mutations
```

Purpose:

> Test whether a model trained on isolated regressions can generalize to multiple defects on the same page.

This is a strong generalization experiment because the model is not explicitly trained on multi-defect combinations.

---

## Experiment 3 — Multi-regression training

```text
Train:
single + multi mutation examples

Test:
multi mutation examples
```

Compare this against Experiment 2.

Research question:

> Does adding multi-defect training data improve detection of simultaneous regressions?

---

## Experiment 4 — Benign differences

Later add benign/no-change examples.

Possible benign examples:

- identical renders;
- cross-browser rendering differences;
- anti-aliasing differences;
- harmless rasterization differences;
- other visual nuisance changes.

Then compare:

```text
Regression-only training
        vs.
Regression + benign training
```

Research question:

> Does training with benign examples reduce false positives without hurting regression recall?

---

# Important note about regression-only training

Right now, all generated mutations are labelled:

```text
REGRESSION
```

There are no benign examples.

This is fine for:

- localization;
- mutation classification;
- feature-difference learning.

However, a binary classifier cannot properly learn:

```text
regression vs benign
```

if it sees only regression examples during training.

So the first phase should be interpreted as:

> Learn visual regression patterns and test how the model behaves when it encounters no-change examples.

A useful result may be that the model performs well on regression localization but produces many false positives on benign pairs.

That can motivate the next phase where benign examples are added.

---

# Model output should support multiple regions

The model should not assume:

```text
"This page contains exactly one defect."
```

Instead, prefer models that produce a **spatial change map**.

Example:

```text
Expected UI
   +
Actual UI
   ↓
Model
   ↓
Change heatmap

┌────────────────────────────┐
│              ███           │
│                            │
│      █████                 │
│                            │
│                     ████   │
└────────────────────────────┘
```

Each changed region can become a separate candidate defect.

For example:

```text
Region 1 → bounding box → typography
Region 2 → bounding box → layout
Region 3 → bounding box → media distortion
```

This is why models such as:

- ChangeFormer;
- BIT;
- DINOv2/DINOv3 spatial features;

are useful.

They can provide spatial outputs rather than only one whole-image label.

---

# Localization pipeline

A practical localization flow could be:

```text
change map
    ↓
threshold
    ↓
connected components
    ↓
region 1
region 2
region 3
    ↓
bounding boxes / masks
    ↓
classify each region
```

The final UI can overlay these boxes or masks on the rendered screenshot.

---

# Classification becomes multi-label / instance-based

For a single mutation, we might predict:

```text
class = font_size
```

For multiple defects, the page may contain:

```json
{
  "regression": true,
  "types": [
    "font_size",
    "alignment",
    "aspect_ratio"
  ]
}
```

A more complete system should eventually operate at several levels.

## Image level

```text
Does this page contain any regression?
```

## Region level

```text
Where are the regressions?
```

## Instance/type level

```text
What is each regression?
```

## Natural-language level

```text
What should the QA engineer know?
```

---

# VLM role in the final system

The visual model should primarily identify:

> **WHERE the change is**

The VLM should primarily explain:

> **WHAT changed**

Example:

```text
Expected screenshot
Actual screenshot
Difference mask / highlighted regions
        │
        ▼
      Qwen3-VL
        │
        ▼
Structured QA description
```

Example prompt:

```text
Image 1: Expected UI
Image 2: Actual UI
Image 3: Difference map

Detected regions:
A, B, C

Describe each meaningful visual regression separately.

Return:
{
  "regression": true,
  "issues": [...]
}
```

Example output:

```json
{
  "regression": true,
  "issues": [
    {
      "region": "A",
      "type": "typography",
      "description": "The heading is smaller than in the reference."
    },
    {
      "region": "B",
      "type": "layout",
      "description": "The CTA is shifted to the right."
    },
    {
      "region": "C",
      "type": "asset",
      "description": "The product image appears horizontally stretched."
    }
  ]
}
```

---

# Visual QA output

The system should ideally produce an annotated screenshot.

Example:

```text
┌──────────────────────────────────────────────┐
│                                              │
│            Example Website                   │
│                                              │
│        ┌───────────────────┐                 │
│        │    GET STARTED    │  ← highlighted │
│        └───────────────────┘                 │
│                                              │
└──────────────────────────────────────────────┘

Regression: YES
Type: Border radius
Confidence: 0.91

"The primary CTA has more rounded corners than the reference."
```

Use:

- bounding boxes;
- translucent masks;
- heatmaps;

rather than requiring the model to literally draw circles.

---

# Real-world QA workflow

A practical deployment could run in CI.

```text
main branch
      ↓
reference screenshots

feature / PR branch
      ↓
new screenshots
      ↓
Regression Lens
      ↓
visual QA report
```

Example workflow:

```text
GitHub Pull Request
        ↓
Build webpage
        ↓
Playwright screenshots
        ↓
Regression Lens
        ↓
Potential regression report
```

Example report:

```text
3 potential visual regressions detected on /checkout:

- Checkout heading font weight differs from reference.
- Pay Now button shifted downward.
- Order-summary card has reduced opacity.
```

The report could include an annotated screenshot showing each region.

---

# Positioning of the project

The safest and most realistic framing is:

> **AI-assisted visual QA**

rather than:

> **fully autonomous QA**

The model flags suspicious differences.

A human QA engineer still decides whether the difference is actually a bug.

This is important because some intentional design changes may look like regressions when compared with an old reference screenshot.

---

# Stronger research question

Instead of only asking:

> How accurately can a model detect single UI mutations?

the project can become:

> **Can models trained on controlled synthetic visual regressions generalize to realistic webpage QA conditions involving unseen webpages, multiple simultaneous defects, and benign visual variation?**

This gives the project a natural progression:

```text
Controlled single defects
        ↓
Multiple simultaneous defects
        ↓
Benign rendering variation
        ↓
Realistic QA scenario
```

---

# Recommended project stages

## Stage 1
Single-regression generation and model comparison.

## Stage 2
Multi-regression test set.

## Stage 3
Multi-regression training experiments.

## Stage 4
Add benign/no-change pairs.

## Stage 5
VLM descriptions.

## Stage 6
Produce annotated QA reports.

This keeps the initial project manageable while giving it a clear path toward a realistic visual-QA application.
