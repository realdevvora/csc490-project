# Regression Lens

**Nuisance-Aware Semantic Visual Regression Detection for Web Interfaces**

CSC490 — Machine Learning for Vision

## Problem Statement

Visual regression testing is still largely a manual process: QA engineers and developers eyeball updated webpages looking for meaningful differences. This is slow, tedious, and error-prone.

The core difficulty is that not every pixel difference is a bug. Browsers can render fonts, images, or anti-aliasing slightly differently even when a page is functioning correctly, while a small shift in a button's position, spacing, font size, opacity, or border radius can represent a genuine regression. A naive pixel-diff approach can't tell these apart.

This project asks a more useful question than "are these two images different?":

> **Is this difference meaningful, where is it, and what changed?**

We propose a model that, given an expected (reference) and actual (rendered) screenshot of the same web page, distinguishes benign rendering variation from real CSS/layout regressions, localizes the change, and classifies what kind of change it is (typography, layout, appearance, shape/chrome). The model is trained on programmatically mutated pairs generated from the [Design2Code](https://github.com/NoviScl/Design2Code) dataset and evaluated against the independent [DiffSpot](https://huggingface.co/datasets/tencent/DiffSpot) benchmark, with the goal of generalizing to web pages it has never seen before.

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

Simple, one-line commit:

```
Update README with data setup instructions
```

Commits with a body for extra context:

```
Filter out mutations with no visible pixel change

Some CSS mutations (e.g. tiny opacity shifts) produced no
detectable difference after rendering. These were polluting
the labeled dataset, so we now discard them during generation.
```

### Pull Requests

- Open a pull request (PR) into `main` when your branch is ready for review.
- Follow the pull request template (to be created)
- Set the pull request to squash commits if you have many small commits that will add lots of clutter to the commit history.
- At least one other team member should review before merging.

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
