# FC-Siam-diff

Per-pixel regression detector for d2c_v3, adapted from Daudt et al., "Fully Convolutional Siamese Networks for Change Detection" (ICIP 2018).

| File | What it is |
|---|---|
| `train_fc_siam_diff.ipynb` | Training, validation and the prediction file |
| `siamunet_diff.py` | Model, unmodified from [rcdaudt/fully_convolutional_change_detection](https://github.com/rcdaudt/fully_convolutional_change_detection) (no license upstream; included with attribution for coursework) |
| `spec.md` | Design decisions, expected results and output format |
| `requirements.txt` | Python dependencies |

## Data
The notebook reads `../../data/processed/d2c_v3/` (gitignored):
```
data/processed/d2c_v3/
  master_dataset_with_benign.json
  splits_85_15.json
  screenshots/
```
Runs are written to `runs/<timestamp>/` in this folder (gitignored): `config.json`, `metrics.jsonl`, `ckpt_last.pt`, `ckpt_best.pt` and `fc_siam_diff_val.jsonl`.

## Run (Linux, NVIDIA GPU)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match nvidia-smi's CUDA version
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# headless, survives SSH disconnects; open the executed notebook afterwards for the outputs
tmux new -s fcsd
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output runs_executed.ipynb train_fc_siam_diff.ipynb
```
Before the full run, set `N_EPOCHS = 2` for a timing run. To resume after an interruption, set `RESUME = 'runs/<id>/ckpt_last.pt'`.

Score the prediction file with the team scorer: `python scripts/score_predictions.py models/fc_siam_diff/runs/<id>/fc_siam_diff_val.jsonl`.
