# Project overview

## Problem

Pixel-level screenshot differences are not equivalent to UI defects. Harmless rasterization, resampling, or browser variation can change pixels, while a small typography or spacing change can violate the intended design. Regression Lens studies whether paired-image vision methods can separate those cases.

The research question is:

> Can a paired-image vision model distinguish meaningful CSS/DOM-level regressions from benign rendering variation, localize the change, and generalize to webpages it has never seen before?

## Intended outputs

Given a reference image and a candidate image of the same webpage, a model should provide:

1. a regression score or binary decision;
2. optional localization, initially represented by a bounding box and potentially a mask; and
3. an optional change category.

Detection is the core task. Localization and categorization are important research outputs, but their exact representation is still open.

## Data strategy

[Design2Code](https://huggingface.co/datasets/SALT-NLP/Design2Code) supplies real webpage HTML and screenshots. The project will create controlled CSS/DOM mutations, render reference/candidate pairs, and store mutation provenance and visual ground truth. The exact upstream revision and imported count must be recorded in a data manifest rather than assumed in code or prose.

[DiffSpot](https://huggingface.co/datasets/tencent/DiffSpot) is reserved for external evaluation after model selection. It is not part of training, hyperparameter tuning, threshold selection, or internal validation. See [ADR 0001](adr/0001-training-and-external-evaluation-datasets.md).

## Research boundaries

- Model architecture and training details are expected to evolve.
- Label, split, metadata, and evaluation contracts should change deliberately because they affect experiment comparability.
- No final model has been selected.
- The first objective is a small, manually inspected, reproducible pipeline—not dataset scale.

Current priorities and unknowns are maintained in [project status](project-status.md), while scope tiers and historical changes are maintained in [scope and roadmap](scope-and-roadmap.md).
