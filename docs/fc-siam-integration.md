# FC-Siam integration — 2026-11-26

Dev's FC-Siam PR #8 is merged into main and is already included in local
`abu-dino-model` at merge commit `1b0f67a`. Fetching found no missing main
commits. This follow-up organizes the imported files into the established
team layout. It does not adopt the separate DiffSpot branch's protocol work.

| Previously under `models/fc_siam_diff/` | Canonical location |
|---|---|
| `siamunet_diff.py` | `src/regression_lens/models/fc_siam_diff.py` |
| `train_fc_siam_diff.ipynb` | `notebooks/fc_siam_diff/` |
| `save_predictions.py` | `scripts/export_fc_siam_predictions.py` |
| `plot_results.py` | `scripts/plot_fc_siam_results.py` |
| `spec.md`, `technical_analysis.md` | `docs/models/fc_siam_diff/` |
| `results/` | `docs/results/fc_siam_diff/` |

Legacy import and command wrappers remain under `models/fc_siam_diff/`.
The small requirements file is retained and has an equivalent `fc-siam`
project extra. New runs default to ignored `outputs/fc_siam_diff/runs/`;
existing ignored run folders are left in place. The lab transfer script now
includes these compatibility files while excluding nested runs/environments.

The notebook locates the repo and split independently of notebook working
directory and accepts `DATA_ROOT` / `FC_SIAM_RUNS_DIR`. Its model import uses
the shared package. The architecture, attribution, parameter names and tensor operations are
preserved; only trailing whitespace in the moved model source was cleaned. Training/evaluation math,
labels, class order, split IDs and curated evidence are unchanged.

The old export script identified training cells by their source-text prefix
and used current notebook settings for any checkpoint. The new entry point
uses explicit notebook cell roles and the checkpoint's saved configuration.
Training and the training data spot-check are excluded. Export remains a
bridge to the notebook's own recipe; it is not a new generic model adapter.
Changing the notebook requires maintaining its export roles. CLI help works
without a dataset/checkpoint. Legacy positional arguments remain accepted.

## Compatibility limits to retain

- Update external notebook/result links to the new locations; the legacy
  Python commands and import remain valid.
- FC-Siam's reported localization has 729/826 regression coverage, uses a
  pixel-diff veto and comes from one run. Preserve these qualifiers beside
  comparisons with DINO or other team models.
- Two failed Firefox renders are omitted from FC-Siam training only. They
  remain in scored subsets; the team's split file is unchanged.
- ConvNeXt/SAM2 shared rescoring remains pending. Folder consistency does
  not make notebook-specific metrics interchangeable with shared scoring.
- The original model source has no upstream license. Its existing coursework
  attribution and licensing note are preserved.

Verification covers model compatibility, exporter selection/configuration,
notebook syntax, CLI help, result integrity and shared rescoring. No new
training, dataset validation, model download, VLM call or DiffSpot benchmark
is part of this organization change.

Verified locally:

- Full suite: **97 passed, 1 skipped** (the optional local `d2c_v2` dataset
  is absent). Pytest used a workspace-local temporary directory because the
  sandbox could not access the existing system pytest temporary directory.
- All saved FC-Siam scorer fields reproduce exactly, excluding only the
  historical prediction source path.
- All eight curated result artifacts and the specification retain their
  original bytes; the model source differs only in trailing whitespace.
- Canonical and legacy exporter help, plotter help and all five chart exports
  work. Relative links in the updated documentation resolve.
- A synthetic checkpoint exported 28 pairs into the correct 13-train /
  15-val subsets, including a zero-score identical pair, without training.
  The real trained checkpoint was not available locally for a new export.
