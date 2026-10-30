# UIClip / CLIP experiment results

Aggregate results of the experiments described in [the UIClip investigation](../../ideas/uiclip-investigation.md), copied from the local `outputs/` folder, which is gitignored. Model weights and feature caches are not committed; see "Not included" below.

## Contents

| Path | What it is |
|---|---|
| `comparison.md` | One row per 10-epoch run (best-AUROC checkpoint), written by `scripts/score_predictions.py` |
| `metrics/<run>__best_auroc.metrics.json` | The scorer's full metrics for each row of `comparison.md` |
| `run_history/<run>/metrics.json` | Every epoch's train/validation loss and validation metrics, plus which epoch was best by AUROC and by F1 |
| `run_history/<run>/config.json` | The run's exact settings, git commit, PyTorch version and GPU |
| `predictions/uiclip_cal_ft4_10_epochs_s*__best_auroc.val_predictions.jsonl` | Validation predictions of the best model (UIClip, last 4 blocks fine-tuned), in the [shared prediction format](../../prediction-format.md) |

## Runs

The batch from `scripts/run_uiclip_experiments.sh`, all validated on the placeholder split (73 pages, 1,108 pairs):

- `uiclip_cal`, `uiclip_nocal`, `clip_cal`, `uiclip_cal_nobenign`, `uiclip_cal_ft4`;
- each trained for 10 epochs, with seeds 0, 1 and 2 (`_10_epochs_s<seed>`).

Two exploratory 5-epoch runs are also included, `uiclip_frozen_s1_5_epochs` and `uiclip_cal_s1_5_epochs`. These are the runs behind the identical-pair finding.

## Reproducing

The 10-epoch runs record commit `3ee96d4` on a clean working tree. That commit was later rewritten to change only its message, and its file contents are identical to `ca409c2` on `abdullah-clip-model-evaluation`. The 5-epoch runs predate the commits and ran on uncommitted code (`"dirty": true`).

To reproduce, check out `ca409c2`, then:
1. unzip the dataset to `d2c_v2/`;
2. run `scripts/run_uiclip_experiments.sh`.

The placeholder split is regenerated identically, because it is seeded. Rescore any predictions with:

```bash
python scripts/score_predictions.py docs/results/uiclip/predictions/*.jsonl --names ft4_s0 ft4_s1 ft4_s2
```

## Not included

- **Model weights** (`outputs/<run>/best_*/model.pt`): 6.6 MB per frozen-encoder run and 115 MB per fine-tuned run. The fine-tuned ones exceed GitHub's 100 MB file limit. Share them outside the repo, e.g. on the team's Google Drive.
- **Feature caches** (`data/processed/features/`): about 49 GB each. Rebuild them with `--cache-features`.
- **Training logs** (`outputs/logs/`).
