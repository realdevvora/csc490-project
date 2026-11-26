# Experiment protocol

## Purpose

This protocol keeps model comparisons meaningful while the implementation evolves. Exact metrics may change, but central changes must be recorded in this file and, when consequential, in an ADR.

## Data partitions

The committed team split is 390 training and 69 validation source pages,
with 25 excluded pages. No internal test partition exists. Preserve the
manifest and label checkpoint/threshold selection as validation-selected.
See [DINO run controls](results/dino/head-runs.md).

- Build train and validation partitions from Design2Code-derived data. An internal test partition is optional (ADR 0005).
- Partition by source webpage, never by screenshot pair.
- Train models, tune hyperparameters and select each model's checkpoint only with Design2Code training and validation data.
- Never train or tune on DiffSpot. Freeze all four Design2Code-selected configurations,
  thresholds, preprocessing, box rules and shared Kimi settings before full-release evaluation.
- Test 1 measures base detection/localization and applicable mapped type metrics.
- Test 2 uses Kimi K2.5 for verification/descriptions, preserving proposal coordinates.
  Its prediction fields, judge reducer and metrics follow pinned official DiffSpot code.
- Keep all native operators and no-diff controls. Only supplementary closed-set type
  scoring may use an explicitly mapped subset. Detector-only scores are not official
  description scores. See [ADR 0007](adr/0007-frozen-full-diffspot-benchmark.md).
- Any change informed by DiffSpot results starts a new declared exploratory round;
  it no longer qualifies as untouched external evaluation.

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

- the same applicable detection/localization metrics on the full frozen DiffSpot release, labelled detector-only or official description scoring; and
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
