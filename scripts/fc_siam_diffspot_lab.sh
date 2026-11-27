#!/usr/bin/env bash
# One command on the lab machine: set up the environment once, then run the
# FC-Siam-diff DiffSpot benchmark. Rerun the same command to continue after any stop.
#
#   bash scripts/fc_siam_diffspot_lab.sh pilot runs/20261111-210302/ckpt_best.pt   # 44 pairs, about $0.10
#   bash scripts/fc_siam_diffspot_lab.sh full  runs/20261111-210302/ckpt_best.pt   # all 4,400 pairs
#
# Needs OPENROUTER_API_KEY=... in .env at the repo root. TORCH_INDEX picks the CUDA wheels.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${VENV:-.venv-diffspot-lab}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}
LOG=outputs/diffspot/run.log

[ -f .env ] || { echo "Put OPENROUTER_API_KEY=... in $(pwd)/.env first"; exit 1; }
[ $# -ge 2 ] && { [ "$1" = pilot ] || [ "$1" = full ]; } && [ -f "$2" ] || {
  echo "Usage: bash $0 pilot|full path/to/ckpt_best.pt [options]"; exit 1; }
[ -f "$(dirname "$2")/config.json" ] || { echo "config.json must sit next to the checkpoint"; exit 1; }

# Install once; the marker file means a half-finished install is redone
if [ ! -f "$VENV/.installed" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install torch --index-url "$TORCH_INDEX"
  "$VENV/bin/pip" install -e ".[diffspot,evaluation,fc-siam]"
  touch "$VENV/.installed"
fi
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p "$(dirname "$LOG")"
"$VENV/bin/python" -u scripts/fc_siam_diffspot_run.py "$@" 2>&1 | tee -a "$LOG"
