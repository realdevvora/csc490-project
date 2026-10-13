# Regression Lens — Model Investigation Plan

## Goal

Compare several visual models for detecting/localizing UI regressions, then use a VLM to describe the detected change.

```text
Expected UI ─┐
             ├── Visual model ──> heatmap / mask / score
Actual UI ───┘

Expected UI + Actual UI + Difference Map
                    │
                    ▼
                   VLM
                    │
                    ▼
        Regression type + description
```

## Team model assignments

### 1. Abu — DINOv3 + DINOv2

**Primary:** DINOv3 ViT-B/16  
**Comparison:** DINOv2 ViT-B/14

Use both as frozen feature extractors:

```text
Expected UI ──> DINO ──> Features A
                            │
                            ├── |A - B| ──> classifier / heatmap
                            │
Actual UI ────> DINO ──> Features B
```

Main tasks:
- extract spatial features;
- compare expected vs actual features;
- generate a change heatmap;
- predict regression type/score;
- highlight the changed region.

**Research question:**  
Does DINOv3 improve fine-grained UI regression detection compared with DINOv2?

---

### 2. Team Member — ChangeFormer V6

Purpose-built before/after change-detection model.

Main tasks:
- input expected + actual screenshots;
- output a pixel-level change mask;
- convert the mask to a highlighted region/bounding box.

**Research question:**  
How well does a dedicated change-detection transformer transfer to UI regression detection?

---

### 3. Team Member — BIT

**BIT — Bi-Temporal Image Transformer**

A simpler established change-detection model.

Main tasks:
- train on expected/actual screenshot pairs;
- generate change masks;
- evaluate detection + localization.

**Research question:**  
How does a simpler bi-temporal change detector compare with ChangeFormer and foundation-model features?

**Stretch:** simple Siamese CNN baseline.

---

### 4. Team Member — UIClip / SigLIP

**Primary:** UIClip  
**Fallback/comparison:** SigLIP Base Patch16-512

```text
Expected UI ──> UIClip ──> Embedding A
Actual UI ────> UIClip ──> Embedding B

[A, B, |A-B|] ──> classifier
```

**Research question:**  
Do UI-aware or semantic representations detect UI regressions better than generic visual features?

---

## Team comparison

| Track | Model | Main idea |
|---|---|---|
| Abu | DINOv3 + DINOv2 | Vision foundation features |
| Member 2 | ChangeFormer V6 | Modern dedicated change detection |
| Member 3 | BIT | Simpler bi-temporal change detection |
| Member 4 | UIClip / SigLIP | UI-aware / semantic features |

Use the same:
- training data;
- validation split;
- DiffSpot evaluation;
- metrics.

---

# VLM stage

Do **not** make a VLM one person's only main model.

First identify the strongest visual detector, then attach a VLM to describe the detected regression.

## Recommended first choice

**Qwen3-VL-4B or 8B**

Alternatives:
- Gemma 3 4B
- InternVL3
- LLaVA-OneVision

### VLM input

```text
Image 1: Expected UI
Image 2: Actual UI
Image 3: Difference map / highlighted regression
```

### Prompt

```text
Identify the meaningful UI regression.

Return:
{
  "regression": true,
  "type": "...",
  "description": "..."
}
```

The visual model finds **where** the regression is.  
The VLM explains **what** changed.

---

# Highlighting the regression

Each visual model should ideally output one of:
- heatmap;
- mask;
- bounding box;
- highlighted region.

Preferred final demo:

```text
Expected UI
Actual UI with regression highlighted

Regression: YES
Type: Border radius
Description:
"The primary CTA has more rounded corners than the reference."
```

---

# Current dataset situation

Right now, all generated CSS/DOM mutations are labelled:

```text
REGRESSION
```

There are currently **no benign training examples**.

That is acceptable for the first experiment.

## Phase 1 — Regression-only training

Train on regression examples only.

Focus on:
- change localization;
- mutation classification;
- feature differences.

Then evaluate on DiffSpot, including no-change examples.

Important:

A binary classifier cannot properly learn:

```text
regression vs benign
```

if it only sees regression examples during training.

So this phase is useful for measuring how badly the model over-predicts regressions on no-change examples.

## Phase 2 — Add benign examples

Later add:
- identical renders;
- cross-browser differences;
- harmless rendering/rasterization differences;
- other nuisance/no-change pairs.

Then compare:

```text
Regression-only training
        vs.
Regression + benign training
```

This could become one of the strongest experiments in the project.

---

# Recommended order

1. Prepare one common train/validation/evaluation pipeline.
2. Each member implements their assigned model.
3. Evaluate all models with the same metrics.
4. Select the strongest 1–2 visual models.
5. Attach a VLM for regression descriptions.
6. Add benign examples and compare against regression-only training.

Suggested metrics:
- precision;
- recall;
- F1;
- false-positive rate;
- localization IoU / pointing accuracy;
- mutation-type Macro-F1.

---

# Final assignment

```text
Abu
DINOv3 ViT-B/16
+ DINOv2 ViT-B/14 comparison

Member 2
ChangeFormer V6

Member 3
BIT

Member 4
UIClip
+ SigLIP fallback/comparison

Team
Best visual detector
+ Qwen3-VL explanation stage
```

This gives the project four distinct model directions while keeping the final system coherent.
