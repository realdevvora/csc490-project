# Experiment protocol

## Purpose

This protocol keeps model comparisons meaningful while the implementation evolves. Exact metrics may change, but central changes must be recorded in this file and, when consequential, in an ADR.

## Data partitions

- Build train, validation, and internal-test partitions from Design2Code-derived data.
- Partition by source webpage, never by screenshot pair.
- Tune models, thresholds, preprocessing, and hyperparameters only with training and validation data.
- Use the internal test partition for the final synthetic-data comparison.
- Evaluate the selected, frozen system on DiffSpot without using DiffSpot to retrain, select checkpoints, tune thresholds, or choose preprocessing.

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

- the same applicable detection/localization metrics on DiffSpot after model selection; and
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
