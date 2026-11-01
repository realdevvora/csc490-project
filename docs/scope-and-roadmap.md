# Scope and roadmap

This file separates commitments from ideas and preserves scope history. When direction changes, update the current sections and append a row to the history table; do not erase the old decision. Use an ADR when the change affects data compatibility, labels, splits, dataset roles, evaluation, or the primary research question.

## Current committed scope

- Investigate paired-image visual regression detection for webpages.
- Generate labelled reference/candidate pairs from Design2Code HTML using controlled mutations.
- Generate benign/no-change examples separately from regression mutations.
- Record mutation provenance, render configuration, localization, and source-page identity.
- Use source-page-disjoint train, validation, and internal-test splits.
- Establish simple baselines before comparing larger candidate models.
- Evaluate the selected frozen system on DiffSpot as an external benchmark.
- Preserve enough configuration and environment information to reproduce serious experiments.

## Probable extensions

- nuisance-aware DINOv2 or ChangeFormer variants;
- pixel masks in addition to target bounding boxes;
- richer benign cross-browser or rasterization variations;
- human-labelled severity or acceptability;
- natural-language descriptions of detected regressions; and
- HTML-email evaluation if webpage results and time permit.

These are not required for minimum project success and should not shape the first pipeline unless promoted through an explicit scope decision.

## Explicit non-goals for the initial project

- training a large vision-language model from scratch;
- production SaaS deployment;
- Outlook/Gmail rendering farms;
- automatic browser-agent QA;
- a full Figma-to-HTML comparison product;
- a general-purpose mutation plugin marketplace;
- a database or distributed orchestration platform; and
- scaling to the full source dataset before the pilot is inspected and trusted.

## Change procedure

For a proposed scope or direction change:

1. State the current assumption and the proposed replacement.
2. Link the evidence: experiment, paper, benchmark limitation, schedule, or engineering constraint.
3. Describe effects on existing data, metadata, splits, and result comparability.
4. Create or supersede an ADR for consequential decisions.
5. Append the change below and update `project-status.md`; do not rewrite prior rows.

## Scope-change history

| Date | Change | Reason | ADR / issue |
|---|---|---|---|
| 2026-10-06 | Established the documented bootstrap scope: webpage pairs from Design2Code, source-page-disjoint internal evaluation, and DiffSpot-only external evaluation. | Create a reproducible baseline before data or model work expands. | ADR 0001–0003 |
