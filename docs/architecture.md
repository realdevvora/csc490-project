# Architecture

## Status

This document describes the intended boundaries for the first trustworthy end-to-end pipeline. It is not a claim that those components are implemented. The current code contains only shared metadata, mutation, and render-configuration contracts.

## Conceptual pipeline

```text
Design2Code source webpage
        |
        v
Mutation selection
        |
        v
HTML/CSS mutation + provenance record
        |
        v
Playwright reference/candidate rendering
        |
        v
Visibility, asset, and quality validation
        |
        v
Structured metadata + page-level split
        |
        v
Candidate models
        |
        v
Internal evaluation and model selection
        |
        v
Frozen model and decision threshold
        |
        v
DiffSpot external evaluation
```

## Separation of concerns

| Area | Responsibility | Must not decide |
|---|---|---|
| Data adapters | Acquire/version source records and load generated pairs | Mutation label semantics |
| Mutation generation | Select a target and apply one controlled change with provenance | Dataset split assignment |
| Rendering | Produce deterministic screenshots from explicit configuration | Whether a visible change is acceptable |
| Validation | Reject missing, invisible, dimension-mismatched, or clearly corrupted pairs | Model predictions |
| Metadata | Preserve source, mutation, render, label, and localization facts | Experiment-specific transforms |
| Models | Consume reference/candidate images and emit task predictions | External benchmark composition |
| Evaluation | Compute versioned metrics using frozen splits and thresholds | Training-time tuning on DiffSpot |
| Experiment configuration | Bind data, model, seed, and environment for a run | Unrecorded global defaults |

These boundaries should remain lightweight. They do not require a plugin framework, service architecture, database, or deep class hierarchy.

## Stable contracts in the scaffold

- `MutationRecord` records mutation family/type, target, before/after values, and parameters.
- `Mutation` is the minimal interface for future mutation implementations.
- `RenderConfig` makes browser and viewport assumptions explicit.
- `GeneratedSample` validates the core pair metadata and relative artifact paths.

The contracts are intentionally narrower than the full conceptual schema in [data design](data-design.md). Extend them only when an implementation needs the additional field.

## Reproducibility boundary

A generated example should be reproducible from:

```text
upstream dataset revision + source page
+ mutation record + random seed
+ rendering configuration and environment
```

The render manifest should eventually include the browser and Playwright versions, viewport, device scale factor, relevant operating-system details, installed fonts, asset-loading outcome, and seed. A configuration file alone is insufficient if it does not identify the software and data revisions used.

## Quality-gate boundary

Generation success does not imply sample acceptance. Before a pair enters a versioned dataset, validation should be able to reject it when:

- either screenshot is missing or cannot be decoded;
- HTML rendering or required asset loading failed;
- the two screenshots unexpectedly have different dimensions;
- no visible pixels changed;
- the intended target cannot be located;
- the changed region is implausibly far outside the intended target; or
- an unrelated page failure dominates the pair.

The bootstrap does not implement a quality score. The first renderer milestone should implement only the smallest checks needed to make manual inspection reliable.
