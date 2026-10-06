# Project status

**Last updated:** 2026-10-06  
**Phase:** Repository bootstrap / data-pipeline design

## What exists

- project proposal and bootstrap brief;
- project and collaboration overview in the root README;
- conceptual data, architecture, evaluation, scope, and decision documentation;
- lightweight Python contracts for mutation provenance, rendering configuration, and generated-sample metadata;
- an example mutation/render configuration and foundational tests; and
- an exploratory Colab mutation notebook on `zehao-data-mutation-pipeline`, not yet integrated into `main`.

No dataset acquisition, Playwright renderer, mutation operator, quality-gate implementation, baseline, training loop, or DiffSpot adapter exists on `main` yet.

## Confirmed decisions

- The primary domain is web interfaces.
- Design2Code-derived pairs are used for training and internal evaluation.
- DiffSpot remains an independent external generalization benchmark.
- Intentional visible CSS/DOM mutations are regressions.
- Benign/no-change examples are generated separately.
- Splits are disjoint by source webpage.
- No final model has been selected.

## Open questions

- Which single mutation type should be implemented first?
- What selector strategy produces controlled, representative targets?
- How many accepted mutations per source page are useful after filtering?
- Which benign variations are realistic, reproducible, and genuinely non-regressive?
- Should localization use bounding boxes, masks, or both in the first dataset version?
- What is the first stable change-category taxonomy?
- How should asset-loading failures and web-font availability be detected?
- Which candidate models will individual team members own?
- Is natural-language regression description useful enough to remain in scope?

## Blockers

None for the next pilot. Before data generation, the team must select and record an exact Design2Code revision and confirm how its HTML/assets are made available to the renderer.

## Next milestones

1. Render clean references for a fixed, manually selected set of 10–20 Design2Code pages.
2. Implement one controlled mutation with complete metadata and visible-change validation.
3. Freeze a page-level split manifest for the pilot and manually review every accepted pair.
4. Run a simple pixel-difference or SSIM baseline only after the pilot data is trusted.
