# FC-Siam-diff workflow

Dev's FC-Siam-diff pipeline is integrated with the shared project layout.
The training recipe, model state-dict keys and saved validation evidence are
preserved. Start the notebook from the repository root or this directory.

| Purpose | Location from the repo root |
|---|---|
| Training and validation | `notebooks/fc_siam_diff/train_fc_siam_diff.ipynb` |
| Original model implementation | `src/regression_lens/models/fc_siam_diff.py` |
| Checkpoint export | `scripts/export_fc_siam_predictions.py` |
| Diagnostic charts | `scripts/plot_fc_siam_results.py` |
| Specification / analysis | `docs/models/fc_siam_diff/` |
| Versioned results | [Predictions, metrics and plots](../../docs/results/fc_siam_diff/README.md) |
| New local runs | `outputs/fc_siam_diff/runs/<timestamp>/` (ignored) |

The architecture comes unmodified from
[Daudt et al.'s upstream implementation](https://github.com/rcdaudt/fully_convolutional_change_detection).
The upstream repository has no license; Dev included it with attribution for
coursework. Moving the file does not change that limitation.

## Environment and data

Use the existing GPU environment when available; do not reinstall a working
PyTorch build just for the folder change. From the repository root on the lab:

```bash
export LAB_ROOT="/virtual/.cache/$USER/csc490"
source "$LAB_ROOT/.venv/bin/activate"
python3 -m pip install -e '.[fc-siam]'
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
export DATA_ROOT="$LAB_ROOT/data/d2c_v3"
export FC_SIAM_RUNS_DIR="$LAB_ROOT/csc490-project/outputs/fc_siam_diff/runs"
```

`DATA_ROOT` must contain `master_dataset_with_benign.json` and its screenshots.
It defaults to `data/processed/d2c_v3`. The committed split remains
`data/splits/splits_85_15.json`: 390 train / 69 val pages, with 25 exclusions.
Two failed Firefox renders are omitted only during training, as in Dev's run;
they are included in exported scoring subsets. No dataset is moved into Git.

On Windows use `.\.venv\Scripts\python.exe` for the install command and set
`$env:DATA_ROOT` to the local dataset directory. The notebook's existing
`NUM_WORKERS = 12` is intended for Linux; use `0` for Windows notebook execution.
The exporter selects zero workers on Windows automatically.

## Train or resume

Open `train_fc_siam_diff.ipynb` and select the intended environment. Configuration
stays in its first code cell. A two-epoch timing run requires changing `N_EPOCHS`
to `2`; it is not the published 50-epoch recipe. For headless lab execution:

```bash
mkdir -p outputs/fc_siam_diff
python3 -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir outputs/fc_siam_diff --output training_executed.ipynb \
  notebooks/fc_siam_diff/train_fc_siam_diff.ipynb
```

Run this inside your usual persistent lab session. Executed notebooks and
checkpoints stay in ignored outputs. To resume, set `RESUME` to an existing
`ckpt_last.pt` path, preferably absolute. Older checkpoints under
`models/fc_siam_diff/runs/` remain where they are; the cleanup does not move
ignored run directories.

## Export and score

Export uses the notebook's setup and prediction definitions, restores
`config.json` beside the checkpoint, and skips training and the training
data spot-check. It requires the full D2C metadata because it exports all,
train and val files. A missing saved configuration stops export; `--config`
can select the correct saved configuration explicitly.

```bash
python3 scripts/export_fc_siam_predictions.py \
  outputs/fc_siam_diff/runs/<id>/ckpt_best.pt \
  --data-root "$DATA_ROOT"
python3 scripts/score_predictions.py \
  outputs/fc_siam_diff/fc_siam_diff_val.jsonl \
  --data-root "$DATA_ROOT" --subset val --names fc_siam_diff \
  --out outputs/fc_siam_diff/scored
python3 scripts/plot_fc_siam_results.py \
  outputs/fc_siam_diff/fc_siam_diff_val.jsonl "$DATA_ROOT"
```

Use `--help` for optional output, data, split and worker arguments. Without a
checkpoint argument the exporter searches the new runs directory and legacy
runs directories for the newest timestamp. Prefer an explicit checkpoint in
reproducible experiments. The old `models/fc_siam_diff/save_predictions.py`
and `plot_results.py` commands and `siamunet_diff.SiamUnet_diff` import forward
to the new files. Their positional checkpoint/output arguments still work.
The moved notebook and `results/` paths must be updated in external bookmarks.

To rescore the committed validation evidence, replace the prediction path with
`docs/results/fc_siam_diff/fc_siam_diff_val.jsonl`. Train scores describe training
pages, not generalization. Keep validation threshold selection and the
localization-coverage caveat in the [results notes](../../docs/results/fc_siam_diff/README.md).
