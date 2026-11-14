# Project status

**Last updated:** 2026-11-14

**Phase:** Shared evaluation / DINO dataset-fit exploration

## What exists

- project, architecture, data, scope, and ADR documentation;
- shared metadata loading, a committed 390/69 train/validation page split, and the 25-page exclusion rule in ADR 0004;
- shared prediction JSONL format, scoring code, and tests;
- model investigations on separate team branches; and
- the first five cells of `notebooks/dino_exploration.ipynb`: environment readout, deterministic matched-sample selection, frozen 448 x 448 crop geometry, and one model-input preview.

The team has completed general dataset validation and manual inspection. The DINO notebook starts from those results. At this checkpoint no DINO weights are loaded, no feature heatmaps have been produced, and no head is trained. A shared DiffSpot adapter is still pending.

## Confirmed decisions

- The primary domain is web interfaces.
- Design2Code-derived pairs are used for training and internal evaluation.
- DiffSpot remains an independent external generalization benchmark.
- Intentional visible CSS/DOM mutations are regressions.
- Benign/no-change examples are generated separately.
- Splits are disjoint by source webpage.
- No final model has been selected.

## Open questions

- Which mutation examples produce concentrated versus diffuse DINO patch-feature responses?
- How do regression distances compare with matched benign and no-change controls?
- How much do the 14 px versus 16 px patch sizes affect localization?
- Does original-scale 448 x 448 crop inference run quickly enough locally for this sample?
- Does DINOv3 improve the same examples after access to its weights is granted?

## Blockers

No dataset-location blocker: local metadata and screenshots are under `D:\csc409-data\d2c_v3`. DINOv3 weight access is not yet available; DINOv2 can proceed first. Lab runtime and CUDA availability will be inspected when moving beyond local CPU inference.

## Next milestones

1. Run and understand the first notebook checkpoint: 13 matched trios (39 pair/crop evaluations), selected from official training pages.
2. Inspect frozen DINOv2 spatial tokens and cosine/L2 heatmaps for one trio, retaining raw distance statistics.
3. Run the fixed sample and record tentative operator-specific strengths and weaknesses.
4. Repeat the same samples and crops with DINOv3; reserve head training and external evaluation for later work.

The crops are guided by known mutation locations. These observations study local feature sensitivity and must not be reported as full-page detection performance. The sample/crop manifest and local artifacts live under ignored `outputs/dino_fit/`. See the [notebook guide](../notebooks/README.md).
