# ADR 0001: Training and external-evaluation datasets

- **Status:** Accepted
- **Date:** 2026-10-06
- **Owners:** Regression Lens team
- **Supersedes:** None
- **Related:** [Project overview](../project-overview.md), [experiment protocol](../experiment-protocol.md)

## Context

The project needs controlled labels for training and an independent measure of generalization. Using the same benchmark for optimization and final reporting would weaken claims about performance on unseen webpages and change distributions.

## Decision

Use Design2Code webpages to generate training, validation, and internal-test data. Reserve DiffSpot as an independent external evaluation benchmark after model and threshold selection.

DiffSpot must not be used for training, checkpoint selection, preprocessing selection, hyperparameter tuning, threshold tuning, or iterative model choice. Training on DiffSpot later requires a new ADR that explicitly supersedes this decision and distinguishes the resulting research claim.

## Rationale

Design2Code source HTML enables controlled mutations with known provenance and localization. DiffSpot provides a different, independently created distribution for assessing cross-dataset generalization. Keeping their roles separate reduces benchmark leakage.

## Alternatives considered

- **Train and evaluate on DiffSpot splits:** offers direct benchmark optimization but weakens the intended cross-dataset generalization test.
- **Combine both datasets before splitting:** increases sample count but removes the clean external benchmark.
- **Use only generated Design2Code data:** supports controlled internal experiments but provides no independent evaluation.

## Consequences

### Positive

- External results measure transfer beyond the generated training distribution.
- The training-data provenance and label policy remain under team control.
- DiffSpot is protected from repeated tuning.

### Negative / trade-offs

- Domain mismatch may reduce external performance.
- The team must design meaningful internal splits and avoid learning generator artifacts.
- DiffSpot-specific failure analysis cannot feed back into the same reported evaluation without declaring another evaluation round.

## Validation / evidence

This decision reflects the research question's emphasis on unseen webpages and the availability of HTML in Design2Code for controlled generation. Empirical validation will come from comparing source-page-disjoint internal results with one frozen external evaluation.

## Revisit when

- DiffSpot's license, schema, or contents make it unsuitable for external evaluation;
- another independent benchmark better matches the target task; or
- the research question explicitly changes from cross-dataset generalization to benchmark-specific adaptation.
