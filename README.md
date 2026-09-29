# Regression Lens

**Nuisance-Aware Semantic Visual Regression Detection for Web Interfaces**

CSC490 — Machine Learning for Vision

## Problem Statement

Visual regression testing is still largely a manual process: QA engineers and developers eyeball updated webpages looking for meaningful differences. This is slow, tedious, and error-prone.

The core difficulty is that not every pixel difference is a bug. Browsers can render fonts, images, or anti-aliasing slightly differently even when a page is functioning correctly, while a small shift in a button's position, spacing, font size, opacity, or border radius can represent a genuine regression. A naive pixel-diff approach can't tell these apart.

This project asks a more useful question than "are these two images different?":

> **Is this difference meaningful, where is it, and what changed?**

We propose a model that, given an expected (reference) and actual (rendered) screenshot of the same web page, distinguishes benign rendering variation from real CSS/layout regressions, localizes the change, and classifies what kind of change it is (typography, layout, appearance, shape/chrome). The model is trained on programmatically mutated pairs generated from the Design2Code dataset and evaluated against the independent DiffSpot benchmark, with the goal of generalizing to web pages it has never seen before.

## Team — Regression Lens

- Abuzar Ansari
- Abdullah Safi
- Zehao Fan
- Dev Vora

## Repository Conventions

### Branch Naming

Branches must be prefixed with your name, followed by what you're working on, in **kebab-case**:

```
<name>-<what-you-are-working-on>
```

Examples:

```
abdullah-model
zehao-data-mutation-pipeline
dev-changeformer-baseline
abuzar-diffspot-eval
```

### Commit Message Conventions

Include a meaningful description of the change, with an optional body. Keep the description short and in the imperative mood (e.g. "add", not "added" or "adds"). Use the commit body for additional context when needed.

#### Examples

Simple, one-line commits:

```
Add DINOv2 pair-fusion baseline
Fix bounding box offset in mutation labels
Generate benign cross-browser render pairs with Playwright
Add opacity mutation family
Remove unused ChangeFormer config
Update README with data setup instructions
```

Commits with a body for extra context:

```
Add nuisance-aware training loop

Trains on (reference, benign variation, regression) triplets so
the encoder learns to be invariant to rendering noise while
staying sensitive to real CSS changes.
```

```
Filter out mutations with no visible pixel change

Some CSS mutations (e.g. tiny opacity shifts) produced no
detectable difference after rendering. These were polluting
the labeled dataset, so we now discard them during generation.
```

```
Switch DiffSpot loader to streaming mode

Loading the full dataset into memory was causing OOM errors on
the lab machines. Streaming keeps memory usage flat at the cost
of slightly slower iteration.
```

### Pull Requests

- Open a PR into `main` when your branch is ready for review.
- Reference related issues where applicable.
- At least one other team member should review before merging.

## Project Overview

- **Primary domain:** webpages / web interfaces
- **Optional stretch:** transfer to responsive HTML emails
- **Training data:** [Design2Code](https://github.com/NoviScl/Design2Code) — 484 real webpages with source HTML and screenshots, used to generate controlled CSS mutation pairs
- **Evaluation data:** [DiffSpot](https://huggingface.co/datasets/tencent/DiffSpot) — paired screenshots with labeled fine-grained visual differences, used as an independent external benchmark
- **Baselines:** pixel diff, SSIM, LPIPS, DINOv2 feature distance, ChangeFormer
- **Main extension:** nuisance-aware representation learning, training on (reference, benign variation, true regression) triplets so the model is invariant to harmless rendering noise while remaining sensitive to real regressions

## Installation

> _TODO: Fill in once the environment and dependencies are finalized._

```bash
# TODO: clone the repo
# TODO: create and activate virtual environment
# TODO: pip install -r requirements.txt
```

## Data Setup

> _TODO: Instructions for downloading/generating Design2Code and DiffSpot data, and running the mutation pipeline._

```bash
# TODO: download Design2Code dataset
# TODO: download DiffSpot dataset
# TODO: run mutation generation script (Playwright-based)
```

## Usage

> _TODO: Add examples once training/inference scripts exist._

```bash
# TODO: train a model
# TODO: run evaluation on DiffSpot
# TODO: run inference on a single image pair
```

## Repository Structure

> _TODO: Fill in once the repo layout is settled (e.g. `data/`, `models/`, `scripts/`, `eval/`)._

```
.
├── README.md
├── TODO/
```

## Models

> _TODO: List models under active investigation (DINOv2, ChangeFormer, and variants) and where their code/configs live._

## Evaluation

> _TODO: Document metrics (accuracy, localization IoU, change-type classification accuracy, etc.) and how to reproduce results._
