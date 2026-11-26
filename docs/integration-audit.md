# DINO/main integration audit — 2026-11-25

This PR integrates current main into `abu-dino-model`. It does not merge other
unmerged branches or change labels, split IDs, training recipes or shared scoring.

## Reconciled

- Preserve both dependency groups and main's lazy model imports.
- Fix a real CLIP cache compatibility failure on Transformers 4.57.6: pass
  its required causal-mask argument and normalize tuple/tensor layer outputs.
  Existing cached/uncached CLIP/SigLIP equivalence tests verify the adapter.
- Resolve current docs to accepted ADR 0005 (no DiffSpot training), retaining
  DINO's 390/69 split and ADR 0006's separate box-IoU extension.
- Mark historical adaptation recommendations as unadopted and VLM as a spec.
- Move `src/zehao/*.ipynb` to `notebooks/convnext_sam2/`, preserving training
  code/results; correct the team notebook's repository URL and scoring data root.
- Preserve the distinct early root mutation notebook in `data/scripts/archive/`
  instead of overwriting the current generation notebook.
- Remove/ignore the tracked browser debug log; keep existing CLI paths stable.
- Use python3 in lab launchers, with configurable environment/data locations.

## Remaining compatibility issues

1. **FC-Siam resolved on 2026-11-26:** Dev's PR #8 is merged into main
   and present in this branch. Its files now follow the shared layout; see
   [the follow-up integration record](fc-siam-integration.md). The earlier
   browser-render branch was not imported by this cleanup.
2. **ConvNeXt/SAM2 remain Colab workflows:** Drive paths are user-specific.
   Raw predictions are not committed; notebook quantile-FPR95/fixed-.5 numbers
   are not harmonized shared-scorer numbers. Rescore exported JSONL first.
3. **Historical notebooks resplit data:** `ConvNeXt.ipynb` and `SAM2.ipynb`
   use the old 411/73 split. Use the team notebook for current comparisons.
   Its inference only scans common image height, potentially missing newly
   added bottom strips. This PR flags that limitation instead of changing runs.
4. **Environments differ:** DINO uses torch 2.8/torchvision .23; SAM2's Colab
   workflow additionally installs timm. No global lockfile or GPU training
   compatibility is claimed. CLIP launchers accept `VENV_PATH`; the team
   launcher also accepts `DATA_ROOT`. Default paths remain compatible.
5. **DiffSpot foundation integrated on 2026-11-26:** ADR 0007 supersedes
   ADR 0005. The shared dataset/scoring and smoke workflow now exists; actual
   model benchmarking and Kimi/judge calls remain pending. See the
   [DiffSpot workflow](diffspot-benchmark.md).
6. **Metric coverage differs:** DINO has no type predictions; IoU is post-hoc.
   Reflow pointing targets bbox_after while IoU uses diff_bbox. Preserve seed
   policy, coverage and recipe differences in comparisons.

## Verification boundary

Run combined local tests, CLI help, notebook source/JSON checks, split/scorer
compatibility and artifact checks. No dataset generation, pretrained downloads,
GPU retraining, DiffSpot scoring or VLM calls are part of this integration.
Actual check results belong in the PR; this audit claims no new model experiment.
