# UIClip / CLIP / SigLIP results on the team split

Aggregate results of every UIClip/CLIP/SigLIP experiment, rerun in one batch on the team split. The findings are in [UIClip / CLIP / SigLIP on the team split](../../ideas/uiclip-team-split-findings.md).

The files are copied from the local `outputs/team_split/` folder, which is gitignored. Model weights and feature caches are not committed; see "Not included" below.

The earlier results, on the old placeholder split, are kept unchanged in [`docs/results/uiclip/`](../uiclip/README.md). The two sets used different validation pages, so their numbers aren't directly comparable.

## Contents

| Path | What it is |
|---|---|
| `comparison.md` | One row per 10-epoch run (best-AUROC checkpoint), written by `scripts/score_predictions.py` |
| `metrics/<run>__best_auroc.metrics.json` | The scorer's full metrics for each row of `comparison.md` |
| `run_history/<run>/metrics.json` | Every epoch's train/validation loss and validation metrics, plus which epoch was best by AUROC and by F1 |
| `run_history/<run>/config.json` | The run's exact settings, git commit, PyTorch version and GPU |
| `predictions/*_cal_ft4_10_epochs_s*__best_auroc.val_predictions.jsonl` | Validation predictions of the fine-tuned models (last 4 blocks): UIClip and CLIP for seeds 0–2, SigLIP for seed 0. They follow the [shared prediction format](../../prediction-format.md) |

## Runs

22 runs, all trained for 10 epochs (`_10_epochs_s<seed>`) by `scripts/run_all_experiments.sh`:

- **Seeds 0, 1 and 2:**
  - frozen: `uiclip_cal`, `uiclip_nocal`, `uiclip_cal_nobenign`, `clip_cal`, `siglip_cal`;
  - fine-tuned: `uiclip_cal_ft4`, `clip_cal_ft4`.
- **Seed 0 only:** `siglip_cal_ft4`.

Every run used the team split, `data/splits/splits_85_15.json`:
- **Training:** 390 pages, 5,909 pairs.
- **Validation:** 69 pages, 1,033 pairs: 826 regression, 138 benign and 69 no-change.
- **Excluded:** 25 pages whose `no_change` pair isn't identical ([ADR 0004](../../adr/0004-exclude-pages-with-broken-no-change-pairs.md)).

The batch ran from 2026-11-11 21:42 to 2026-11-12 08:13 (10 h 31 min) on an RTX 5070 Ti. Every feature cache already existed.

## Reproducing

Every `config.json` records commit `c2e9b7b` on `abdullah-clip-model-evaluation`, with a clean working tree (`"dirty": false`).

1. Check out `c2e9b7b`.
2. Unzip the dataset to `d2c_v2/`.
3. Run `scripts/run_all_experiments.sh`. It checks that the split file matches the dataset before it starts.

Rescore the predictions with:

```bash
python scripts/score_predictions.py docs/results/uiclip_team_split/predictions/uiclip_cal_ft4_*.jsonl --names ft4_s0 ft4_s1 ft4_s2
```

The team split is the scorer's default, so no `--split` is needed.

## Not included

- **Model weights** (`outputs/team_split/<run>/best_*/model.pt`): 6.6 MB per frozen-encoder run and 115 MB per fine-tuned run. The fine-tuned ones exceed GitHub's 100 MB file limit. Share them outside the repo, e.g. on the team's Google Drive.
- **Feature caches** (`data/processed/features/`): about 49 GB each for UIClip/CLIP and about 266 GB for SigLIP-512. Rebuild them with `--cache-features`.
- **Training logs** (`outputs/logs/team_split/`).
