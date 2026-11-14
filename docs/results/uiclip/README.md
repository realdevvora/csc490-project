# UIClip / CLIP / SigLIP experiment results

> These are the results on the **old placeholder split**, kept for reference. The current results, with every experiment rerun on the team split, are in [`docs/results/uiclip_team_split/`](../uiclip_team_split/README.md).

Aggregate results of the experiments described in [the UIClip investigation](../../ideas/uiclip-investigation.md), copied from the local `outputs/` folder, which is gitignored. Model weights and feature caches are not committed; see "Not included" below.

## Contents

| Path | What it is |
|---|---|
| `comparison.md` | One row per 10-epoch run (best-AUROC checkpoint), written by `scripts/score_predictions.py` |
| `metrics/<run>__best_auroc.metrics.json` | The scorer's full metrics for each row of `comparison.md` |
| `run_history/<run>/metrics.json` | Every epoch's train/validation loss and validation metrics, plus which epoch was best by AUROC and by F1 |
| `run_history/<run>/config.json` | The run's exact settings, git commit, PyTorch version and GPU |
| `predictions/*_cal_ft4_10_epochs_s*__best_auroc.val_predictions.jsonl` | Validation predictions of the fine-tuned models (last 4 blocks): UIClip and CLIP for seeds 0–2, SigLIP for seed 0. They follow the [shared prediction format](../../prediction-format.md) |

## Runs

All runs were trained for 10 epochs (`_10_epochs_s<seed>`) and validated on the **old placeholder split** (73 pages, 1,108 pairs). They were not run on the team split (`data/splits/splits_85_15.json`), which shares no validation pages with it:

- **First batch** (`scripts/run_uiclip_experiments.sh`), seeds 0, 1 and 2: `uiclip_cal`, `uiclip_nocal`, `clip_cal`, `uiclip_cal_nobenign`, `uiclip_cal_ft4`.
- **Follow-up batch** (`scripts/run_clip_siglip_experiments.sh`):
  - `siglip_cal` and `clip_cal_ft4`, seeds 0, 1 and 2;
  - `siglip_cal_ft4`, seed 0 only.

Two exploratory 5-epoch runs are also included, `uiclip_frozen_s1_5_epochs` and `uiclip_cal_s1_5_epochs`. These are the runs behind the identical-pair finding.

## Reproducing

Commits recorded in each `config.json`:

- **First batch:** a clean working tree at `ca409c2` on `abdullah-clip-model-evaluation`.
  - These runs actually used an earlier copy of that commit whose message was later rewritten. Its files are identical, so its hash in the configs was replaced with `ca409c2`.
- **Follow-up batch:** `0fae2d5`, marked `"dirty": true`.
  - The only uncommitted file was the new `scripts/run_clip_siglip_experiments.sh` itself. The training code is exactly that commit.
- **The 5-epoch runs:** these predate the commits and ran on uncommitted code (`"dirty": true`).

To reproduce:
1. Check out the commit recorded in the run's config.
2. Unzip the dataset to `d2c_v2/`.
3. Run the batch script: `scripts/run_uiclip_experiments.sh` or `scripts/run_clip_siglip_experiments.sh`.

At those commits the training script regenerates the placeholder split identically, because it is seeded. The scripts now default to the team split, so rescoring these predictions needs the placeholder passed explicitly. It's kept locally as `data/processed/DO_NOT_USE_splits_placeholder_85_15.json`; if it's missing, recreate it with `regression_lens.data.splits.make_page_split`:

```bash
python scripts/score_predictions.py docs/results/uiclip/predictions/uiclip_cal_ft4_*.jsonl --names ft4_s0 ft4_s1 ft4_s2 \
    --split data/processed/DO_NOT_USE_splits_placeholder_85_15.json
```

## Not included

- **Model weights** (`outputs/<run>/best_*/model.pt`): 6.6 MB per frozen-encoder run and 115 MB per fine-tuned run, for every backbone. The fine-tuned ones exceed GitHub's 100 MB file limit. Share them outside the repo, e.g. on the team's Google Drive.
- **Feature caches** (`data/processed/features/`): about 49 GB each for UIClip/CLIP and about 266 GB for SigLIP-512. Rebuild them with `--cache-features`.
- **Training logs** (`outputs/logs/`).
