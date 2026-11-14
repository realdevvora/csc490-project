# ADR 0005: Split DiffSpot into a model-comparison half and a final-evaluation half

- **Status:** Accepted
- **Date:** 2026-11-13
- **Owners:** Regression Lens team
- **Supersedes:** [ADR 0001](0001-training-and-external-evaluation-datasets.md)
- **Related:** [Experiment protocol](../experiment-protocol.md), [ADR 0003](0003-source-page-disjoint-splits.md), [UIClip / CLIP / SigLIP on the team split](../ideas/uiclip-team-split-findings.md)

## Context

ADR 0001 used Design2Code for training, validation and an internal test split, and reserved DiffSpot for one final evaluation. DiffSpot was never to be used for model choice, checkpoint selection or threshold tuning.

Two things have changed since then:

- **Validation can no longer tell models apart.**
  - On the team split's validation set, every fine-tuned UIClip, CLIP and SigLIP model reaches AUROC ≥ 0.9996. None of them flags a benign pair at the threshold that catches 95% of regressions.
  - Each model's checkpoint (best validation AUROC) and threshold are chosen on those same validation pages, so the numbers are also optimistic.
  - When several epochs tie at AUROC 1.000, the choice among them is arbitrary, and validation can't show whether that matters.
- **The project now has two separate questions to answer:**
  1. Which detector, from the team's investigation tracks, works best on real web pages?
  2. How well does the final integrated system, the chosen detector plus the VLM description stage, work on pages it has never seen?

  ADR 0001 provides no held-out data for the first question. Answering it on the same data as the second would leave no unbiased final number.

There is no internal Design2Code test split. Adding one would require re-splitting and rerunning every model, and it would only measure performance on our own generated distribution.

## Decision

**DiffSpot is divided once into two fixed halves with separate roles.**

| | Comparison half (50%) | Final-evaluation half (50%) |
|---|---|---|
| **Purpose** | Compare the team's detectors and choose one | Evaluate the frozen integrated system (detector + VLM) |
| **Allowed** | Comparing models, choosing the detector, choosing the decision threshold, choosing VLM prompts and post-processing, failure analysis | Running the frozen system and reporting its results |
| **Not allowed** | Training on it | Training, model choice, threshold or prompt tuning, or any change made after looking at its results |
| **Its numbers are** | Optimistic for the chosen model, because it was chosen there | The project's unbiased estimate of real-world performance |

Rules for both halves:

- **No training on DiffSpot.** Neither half is used for training. This is unchanged from ADR 0001.
- **Checkpoints are chosen on Design2Code validation.** Each model's checkpoint comes from its own training data, for example best validation AUROC for the UIClip track. DiffSpot's comparison half compares those already-chosen checkpoints. It doesn't select epochs.
- **The final half is used once.** If anything is changed after looking at it, the next evaluation is declared a new round, and the earlier result is reported as the only unbiased one.
- **The split is decided once and committed.** It is page-disjoint: every pair from the same source page or site goes in the same half, following ADR 0003. It is stratified by DiffSpot's Easy/Medium/Hard levels and generated from a recorded seed. Like `data/splits/splits_85_15.json`, it is committed as a manifest, and it is not changed after either half has been used.

**Design2Code** stays the only training data. It provides the training and validation partitions (the team split). An internal Design2Code test split is no longer required. It may still be added if the team wants an unbiased in-distribution number.

## Rationale

- **It compares models on data none of them were selected on.** This removes the optimism of validation numbers, and it shows whether the arbitrary choice among tied best-AUROC epochs matters. Changing the selection rule could not do either.
- **It compares models on the target distribution.** Real web pages are what the project cares about. An internal Design2Code test split would measure only our own generator's distribution, and the models have already saturated there.
- **It keeps one honest final number.** Model choice and integration decisions use the comparison half, so the final half stays untouched for the integrated system.
- **It's cheap.** Unlike an internal test split, it needs no re-split of Design2Code and no retraining.

## Alternatives considered

- **Keep ADR 0001 and compare models on an internal Design2Code test split:** that keeps DiffSpot fully blind. But it needs a re-split and a rerun of every model, measures only the generated distribution, and is likely to saturate as validation has.
- **Compare and report on all of DiffSpot:** that uses all the data. But choosing the model on the same pages that are reported makes the final number optimistic, and the integrated system would have no held-out evaluation.
- **Compare on Design2Code validation only:** that needs no new data, but the fine-tuned models are already tied there, and the numbers are optimistic.
- **Repeated or cross-validated DiffSpot splits:** these give more stable comparisons, but every page would eventually influence model choice, so no unbiased final number would remain.

## Consequences

### Positive

- Models from different tracks are compared on held-out, real-world pages, under the same rules.
- The integrated system gets an evaluation that no choice was tuned on.
- Each checkpoint rule (for the UIClip track, best validation AUROC) is tested on data it wasn't chosen on.

### Negative / trade-offs

- **Each half is half the size,** so confidence intervals are wider. Small differences between models may not be resolvable.
- **The external claim is weaker than ADR 0001's.** The final half comes from the same distribution the model was chosen on. So it measures performance on unseen pages of a known benchmark, not a fully blind cross-dataset transfer.
- **Comparison-half numbers are optimistic for the chosen model** and must be labelled as such.
- **There's no unbiased number on Design2Code** unless an internal test split is added later.
- **Some metrics may not exist on DiffSpot.** If it has few benign-like pairs, benign FPR can't be measured there. Its change categories won't match our 13 mutation types.

## Validation / evidence

- On the team split, the fine-tuned models' validation AUROC is 0.9996–1.0000, and their benign FPR at 95% recall is 0 ([findings](../ideas/uiclip-team-split-findings.md)). Validation can't rank them.
- 15 of the 22 UIClip-track runs selected epoch 9 or 10 by best AUROC, and for fine-tuned models that choice is among tied epochs. Its effect can only be measured on held-out data.

## Open before first use

- **Adapter:** map DiffSpot's records and labels to the shared prediction format and metrics (`docs/prediction-format.md`).
- **Split manifest:** create and commit it once the size, page or site grouping and difficulty labels are known.
- **Metrics:** confirm which ones DiffSpot supports, especially benign or no-change false-positive rates and change-type categories.
- **Size check:** confirm that each half is large enough to separate models. Report confidence intervals.

## Revisit when

- DiffSpot turns out too small for a half to separate the candidate models;
- another independent benchmark becomes available that better matches the task;
- the team wants to train or fine-tune on DiffSpot. That needs a new ADR and a different research claim; or
- DiffSpot's license, schema or contents make it unsuitable for evaluation.
