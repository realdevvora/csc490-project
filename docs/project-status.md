# Project status

**Last updated:** 2026-11-14

**Phase:** Shared evaluation / DINO dataset-fit exploration

## What exists

- project, architecture, data, scope, and ADR documentation;
- shared metadata loading, a committed 390/69 train/validation page split, and the 25-page exclusion rule in ADR 0004;
- shared prediction JSONL format, scoring code, and tests;
- model investigations on separate team branches; and
- two checkpoints in `notebooks/dino_exploration.ipynb`: a fixed 13-trio sample and crop geometry, followed by frozen DINOv2 inference, token inspection, cosine/L2 heatmaps, and raw statistics for one trio.

The team has completed general dataset validation and manual inspection. The DINO notebook starts from those results. DINOv2 has run locally on the font-size regression and its matched benign/no-change controls. The regression response is localized, but benign browser variation produces larger responses; the no-change L2 distance is zero. These are single-trio preliminary observations, not operator-wide or full-page results. See the [pilot notes](results/dino/README.md). No head is trained. A shared DiffSpot adapter is still pending.

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
- Does the initial local timing (approximately 3 seconds per image) hold across the remaining fixed sample?
- Does DINOv3 improve the same examples after access to its weights is granted?

## Blockers

No dataset-location blocker: local metadata and screenshots are under `D:\csc409-data\d2c_v3`. DINOv3 weight access is not yet available; DINOv2 can proceed first. Lab runtime and CUDA availability will be inspected when moving beyond local CPU inference.

## Next milestones

1. Review checkpoint 2's patch tensors, heatmaps, raw statistics, and tight-box pointing miss for the font-size trio.
2. Run the remaining fixed sample and record tentative operator-specific strengths and weaknesses.
3. Repeat the same samples and crops with DINOv3; reserve head training and external evaluation for later work.

The crops are guided by known mutation locations. These observations study local feature sensitivity and must not be reported as full-page detection performance. The sample/crop manifest and local artifacts live under ignored `outputs/dino_fit/`. See the [notebook guide](../notebooks/README.md).
