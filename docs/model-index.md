# Model and experiment directory

Updated 2026-11-26. Implementations, historical results and proposals have
different maturity levels. No model is declared the overall winner.

| Track | Entry points | Evidence / limitations |
|---|---|---|
| DINOv2/v3 | `scripts/dino_heads.py`, `run_dino_fullpage.py`; `src/regression_lens/models/dino_*.py` | [Verified A/B/C report](results/dino/trained-validation-results.md). Three trained seeds; preserve v1/v2 crop namespaces. Type unavailable; IoU is a separate fixed-rule extension. |
| UIClip/CLIP/SigLIP | `scripts/train_uiclip.py`, `run_all_experiments.sh`; `src/regression_lens/models/uiclip.py` | [Team-split results](results/uiclip_team_split/README.md). Older `results/uiclip/` uses a different split. `run_all_experiments.sh` runs this track, not all team models. |
| ConvNeXt/SAM2 encoder | [Team notebook](../notebooks/convnext_sam2/CSC490_ConvNeXt_SAM2_team.ipynb) | Correct team split reported. Rounded notebook metrics use quantile FPR95/fixed-.5 classification; shared rescoring remains pending. No verified pointing/IoU. SAM2 is an encoder classifier, not a mask decoder. |
| FC-Siam-diff | [Workflow/notebook](../notebooks/fc_siam_diff/README.md); `scripts/export_fc_siam_predictions.py`, `plot_fc_siam_results.py`; `src/regression_lens/models/fc_siam_diff.py` | [Shared-scorer results](results/fc_siam_diff/README.md). 1,033 validation pairs; localization covers 729/826 regressions. Pixel-diff veto and single-run evidence limit direct comparison. |
| VLM/DiffSpot | [Diagnosis specification](vlm-diagnosis-format.md), ADR 0007 | Dataset/scoring foundation now exists; see [workflow](diffspot-benchmark.md) and ADR 0007. No completed model benchmark or paid API runner; adaptation suggestions remain unadopted. |

The shared contract is [prediction JSONL/scoring](prediction-format.md), using
the unchanged 390/69-page split with 25 exclusions. Name checkpoint, threshold
rule, seeds, data revision and localization support in every comparison. DINO
and CLIP-family resolutions/budgets differ; comparisons are not architecture-only
or equal-compute. Coordinates are always original screenshot pixels. Missing
outputs mean unavailable, not zero. Mutation IDs must never enter VLM prompts.

Data-generation notebooks live in `data/scripts/`, model notebooks in
`notebooks/`, reusable importable Python in `src/`, stable CLI entry points in
`scripts/`, curated evidence in `docs/results/`, and local runs in ignored
`outputs/`. FC-Siam legacy imports and CLI names forward to the canonical files; see its workflow for moved notebook/result paths.
