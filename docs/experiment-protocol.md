# Experiment protocol

## Purpose

This protocol keeps model comparisons meaningful while the implementation evolves. Exact metrics may change, but central changes must be recorded in this file and, when consequential, in an ADR.

## Data partitions

- Build train and validation partitions from Design2Code-derived data. An internal test partition is optional ([ADR 0005](adr/0005-diffspot-comparison-and-final-evaluation-halves.md)).
- Partition by source webpage, never by screenshot pair.
- Train models, tune hyperparameters and select each model's checkpoint only with Design2Code training and validation data.
- Never train on DiffSpot. Split it once into two page-disjoint halves, stratified by difficulty:
  - **Comparison half:** compare the team's models and choose the detector, its decision threshold, and the VLM prompts and post-processing. Results here are optimistic for the chosen model and must be labelled as such.
  - **Final-evaluation half:** evaluate the frozen integrated system once. Nothing may be tuned on it; a change made after looking at it starts a new, declared evaluation round.

Every result must name the upstream data revision, generation configuration, generated-dataset version, and split-manifest version.

## Model-level contract

```text
input:
  reference image
  candidate image

outputs:
  regression score
  optional localization
  optional change category
```

Candidate baselines include pixel difference, SSIM, LPIPS, DINOv2-derived paired features, and ChangeFormer. Nuisance-aware variants and other models remain candidates, not commitments. A simpler baseline should be established before expensive models are compared.

## Metrics

### Binary regression detection

- precision;
- recall;
- F1;
- false-positive rate on benign/no-change examples; and
- threshold and threshold-selection method.

Accuracy may be reported but should not replace class-sensitive metrics.

### Localization

- bounding-box intersection over union;
- pointing accuracy; and
- pixel intersection over union if masks are introduced.

### Change category

- macro F1 over the active taxonomy; and
- per-category support and F1.

### Generalization

- the same applicable detection/localization metrics on each DiffSpot half, labelled as comparison or final evaluation; and
- easy/medium/hard breakdowns where supported by the benchmark metadata.

## Repeated runs

Report random seeds and, when feasible, mean and variability across repeated runs. A single exploratory run may be reported as such, but it must not be presented as a stable comparison.

## Run record

Serious experiments should write to an ignored directory such as:

```text
outputs/<experiment-name>/
  config.yaml
  metrics.json
  notes.md
```

The record should include:

- commit SHA and dirty-working-tree status;
- dataset and split versions;
- generation and model configurations;
- random seed;
- relevant package, browser, and hardware information;
- metrics and threshold-selection method; and
- concise observations, failures, and deviations from this protocol.

MLflow or Weights & Biases is not required during the bootstrap phase. The priority is complete, reviewable records rather than a particular tracking service.
