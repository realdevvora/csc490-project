# Regression Lens

**Nuisance-aware semantic visual regression detection for web interfaces**

Regression Lens is a CSC490 Machine Learning for Vision capstone project. Given a reference screenshot and a candidate screenshot of the same webpage, the project investigates whether a paired-image system can:

1. detect a meaningful visual regression;
2. localize the changed region; and
3. classify the kind of UI change.

**Current phase:** shared evaluation tooling and model investigations. This branch is beginning a pretrained DINO dataset-fit study; see the [notebook guide](notebooks/README.md).

## Current direction

- Design2Code webpages will provide source HTML for generated training and internal-evaluation pairs.
- DiffSpot is reserved as an independent external generalization benchmark and is not training or model-selection data.
- Intentional visible CSS/DOM mutations are regressions, regardless of whether the numerical change is small.
- Benign/no-change examples are generated and labelled separately.
- All examples derived from one source webpage belong to one dataset split.
- Candidate approaches remain open: pixel difference, SSIM, LPIPS, DINOv2-based methods, ChangeFormer, and nuisance-aware variants.

These commitments and their rationale are recorded in the [architecture decision records](docs/adr/README.md). Current unknowns and priorities are tracked in [project status](docs/project-status.md).

## Intended pipeline

```text
Design2Code source webpage
        -> controlled HTML/CSS mutation
        -> reproducible Playwright rendering
        -> reference/candidate screenshot pair
        -> visibility and quality checks
        -> structured metadata and page-level split
        -> candidate model training and internal evaluation
        -> frozen model
        -> DiffSpot external evaluation
```

The project proposal is available at [Regression_Lens_CSC490_Project_Report_Final_Illustrated.pdf](Regression_Lens_CSC490_Project_Report_Final_Illustrated.pdf).

## Repository layout

```text
.
|- configs/                       # versioned example configuration
|- data/README.md                 # local-data policy (datasets stay out of Git)
|- docs/                          # architecture, data, evaluation, status, scope, ADRs
|- notebooks/                     # incremental DINO feature exploration
|- src/regression_lens/
|  |- data/                       # generated-pair metadata contract
|  |- mutations/                  # mutation provenance and interface
|  |- rendering/                  # reproducible render configuration
|  |- models/                     # future paired-image model adapters
|  `- evaluation/                 # future metric adapters
|- tests/                         # tests for the stable scaffold contracts
`- .github/pull_request_template.md
```

Generated data, browser artifacts, experiment outputs, and checkpoints are ignored by Git. The repository should contain code, versioned configuration, manifests, aggregate results, and documentation—not screenshot corpora or model weights.

## Setup

Python 3.10 or newer is required. The current package has no runtime dependencies; the development extra installs the tools needed for scaffold validation.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

Playwright and data/model dependencies will be added with the first implementation that uses them. Dataset downloads are intentionally not automated during this bootstrap phase.

## Documentation

- [Project overview](docs/project-overview.md)
- [Architecture](docs/architecture.md)
- [Data design](docs/data-design.md)
- [Experiment protocol](docs/experiment-protocol.md)
- [Project status](docs/project-status.md)
- [Scope and roadmap](docs/scope-and-roadmap.md)
- [Architecture decision records](docs/adr/README.md)
- [DINO exploration setup and checkpoints](notebooks/README.md)

## Team

- Abuzar Ansari
- Abdullah Safi
- Zehao Fan
- Dev Vora

## Collaboration conventions

Use `<name>-<work-item>` in kebab case for branches, for example `abdullah-mutation-pipeline`. Use concise imperative commit summaries and include a body when the decision or experiment context is not obvious from the diff.

Open pull requests into `main`, complete the pull-request checklist, and obtain at least one team review before merging. Changes to dataset roles, label semantics, split strategy, evaluation protocol, or committed scope must update the relevant documentation and usually require an ADR.

## Next milestone

On `abu-dino-model`, study frozen DINOv2 patch-feature differences on a fixed sample of regressions, benign pairs, and no-change controls. Repeat identical crops with DINOv3 after its weight access is available. General dataset validation and manual inspection are already complete; this stage studies model sensitivity before head training.
