# ADR 0007: Freeze four Design2Code-selected pipelines and evaluate all DiffSpot

- **Status:** Accepted
- **Date:** 2026-11-25
- **Owners:** Regression Lens team; approved by the user
- **Supersedes:** [ADR 0005](0005-diffspot-comparison-and-final-evaluation-halves.md)
- **Related:** [DiffSpot workflow](../diffspot-benchmark.md)

## Decision and rationale

Each teammate selects one configuration on Design2Code validation. Freeze the
four checkpoints/seed policies, thresholds, preprocessing, box rules and shared
Kimi K2.5 settings before evaluating the entire pinned DiffSpot release.
Neither training nor model, threshold, prompt or post-processing selection uses
DiffSpot. No comparison/final half split is created. This uses the whole dataset
for an external comparison instead of choosing a single model on half of it.

Test 1 evaluates base detectors on every operator, including novel types.
Test 2 uses the same models and Kimi K2.5 to verify and describe proposed regions;
it may reject proposals but cannot modify their coordinates. Both tests retain
no-diff controls. Only supplementary closed-set type metrics use an exact mapped
subset; do not delete unmatched operators from detection/localization/description.

Test 2 uses the pinned official DiffSpot prediction fields, judge-label reducer
and aggregation, including all six required metrics and per-operator reports.
Detector-only binary metrics are not relabelled as official mutation-identification
scores. Base models may submit genuine descriptive output for official judging;
otherwise official description metrics are unavailable. No ground-truth-derived
caption or oracle box may be supplied to a model.

Published Kimi-only results are external context; no standalone Kimi rerun is
required. They do not establish a paired improvement from detector guidance.
The original Design2Code split, labels and scorer remain unchanged.

## Consequences and evaluation limits

The dataset/scoring foundation prepares data, freezes settings, validates
submitted predictions and consumes external official judge labels. It makes
no Kimi or judge API calls and does not train/reimplement teammates' models.
Actual final comparisons require all four owners' fingerprints and outputs.
Checkpoints are fingerprinted by owners; this tool does not inspect their weights.

If settings change after inspecting DiffSpot outcomes, declare a new exploratory
round. Results no longer represent untouched external evaluation. Prompt and
threshold development must use Design2Code. Preserve native target-element
boxes and use their visible intersection for supplementary localization:
release metadata contains 330 partly off-screen boxes. This does not alter
official description scoring. Keep localization distinct from Design2Code
changed-pixel targets.
With fixed proposal coordinates, Kimi can improve verification/descriptions,
not the geometry of retained boxes. Missing/rejected boxes count as misses in
end-to-end localization; conditional metrics retain explicit coverage.

## Verification

Validate the complete release, including official routing of no-diff controls
with unknown difficulty. Test official reducer/metric parity and strict coverage,
taxonomy handling, leak-free inference exports and changed-box rejection.
Pin upstream scoring code/prompts and preserve their MIT license and hashes.
