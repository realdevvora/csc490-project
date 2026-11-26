# Regression Lens

**Nuisance-aware visual regression detection for web interfaces ? CSC490**

Given expected and actual screenshots, investigate regression detection,
localization and optional change classification. Natural-language diagnosis
is a planned integration stage. No overall winner or deployment claim is made.

## Current work

| Track | Entry point | Results |
|---|---|---|
| DINOv2/v3 | `scripts/dino_heads.py`, `run_dino_fullpage.py` | [Verified A/B/C and box-IoU report](docs/results/dino/trained-validation-results.md) |
| UIClip/CLIP/SigLIP | `scripts/train_uiclip.py`, `run_all_experiments.sh` | [Team-split results](docs/results/uiclip_team_split/README.md) |
| ConvNeXt/SAM2 encoder | [Team notebook](notebooks/convnext_sam2/CSC490_ConvNeXt_SAM2_team.ipynb) | Notebook-reported; shared rescoring pending |
| FC-Siam-diff | [Training/export workflow](notebooks/fc_siam_diff/README.md) | [Shared-scorer results and charts](docs/results/fc_siam_diff/README.md) |

See [model directory](docs/model-index.md), [integration audit](docs/integration-audit.md)
and [project status](docs/project-status.md). The [DINO student guide](docs/dino-experiments-explained.md)
explains its experiments and logs.

## Shared contracts

- Design2Code supplies **390 train / 69 validation pages**, with 25 exclusions,
  in `data/splits/splits_85_15.json`. Never split derivatives of one page apart.
- Intentional visible CSS/DOM mutations are regressions; benign browser and
  identical controls are separate. Small mutations are not automatically benign.
- Use [shared prediction JSONL/scoring](docs/prediction-format.md). Optional
  type/point/box outputs are not guaranteed for every model; unavailable is not zero.
- Never train on DiffSpot. [ADR 0005](docs/adr/0005-diffspot-comparison-and-final-evaluation-halves.md)
  defines comparison and untouched final-evaluation halves. The earlier evaluator
  was reverted on main; no completed DiffSpot benchmark is claimed here.
- [VLM diagnosis format](docs/vlm-diagnosis-format.md) is a specification, not an
  implemented integrated predictor. A four-pipeline comparison remains a proposal.

## Repository layout

```text
configs/                         versioned examples
data/scripts/                    generation notebooks; archive/ preserves history
data/splits/                     committed page manifests; datasets stay local
notebooks/dino_exploration.ipynb  incremental DINO exploration
notebooks/convnext_sam2/          current and historical model experiments
notebooks/fc_siam_diff/           FC-Siam training and workflow
models/fc_siam_diff/              compatibility imports and CLI wrappers
src/regression_lens/             importable model/data/render/evaluation modules
scripts/                         stable train, score and reporting commands
docs/results/                    curated aggregate results and figures
docs/adr/                        consequential decisions and protocol history
tests/                           shared-contract and component checks
outputs/                         ignored local runs, screenshots, caches and weights
```

## Setup and verification

Python 3.10+ is required. The base package does not install model frameworks.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,evaluation]"
.\.venv\Scripts\python.exe -m pytest
```

Additional extras: `dino`, `uiclip`, `fc-siam`, `figures`. ConvNeXt/SAM2 setup is documented
[alongside its notebooks](notebooks/convnext_sam2/README.md). Use separate
model environments when requirements differ. Optional model tests may skip
when dependencies are missing; tests do not require pretrained downloads.

On the lab always use **python3**, including inside its environment. Follow
[lab setup](docs/lab-setup.md) to keep data, environments and caches off the
home quota. Existing CLI names and run namespaces remain valid.

## Documentation and collaboration

- [Experiment protocol](docs/experiment-protocol.md)
- [Data policy](data/README.md) and [data design](docs/data-design.md)
- [Architecture](docs/architecture.md) and [overview](docs/project-overview.md)
- [Scope/history](docs/scope-and-roadmap.md) and [ADRs](docs/adr/README.md)
- [Original proposal](Regression_Lens_CSC490_Project_Report_Final_Illustrated.pdf)

Team: Abuzar Ansari, Abdullah Safi, Zehao Fan, Dev Vora.
Open PRs into `main` and obtain at least one team review before merging.
Record changes to labels, splits, dataset roles and evaluation in docs/ADRs.
Keep datasets, credentials, weights and local runs out of Git. Curated aggregate
figures, metrics and small prediction files remain versioned experiment evidence.
