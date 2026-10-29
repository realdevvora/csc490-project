#!/usr/bin/env bash
# Run the UIClip-track experiments one after another, then compare them.
#
# Each experiment is trained for 10 epochs with seeds 0, 1 and 2, using cached
# encoder features. Runs go to outputs/<experiment>_10_epochs_s<seed>/ and logs
# to outputs/logs/. A failed run is reported and the rest continue. Running the
# script again re-trains and overwrites runs with the same names.
#
# Start it in the background (keeps running if the terminal closes):
#   mkdir -p outputs/logs && nohup bash scripts/run_uiclip_experiments.sh > outputs/logs/all.log 2>&1 &
# Watch progress:     tail -f outputs/logs/all.log
# Watch one run:      tail -f outputs/logs/clip_cal_10_epochs_s0.log
# Stop everything:    pkill -f run_uiclip_experiments.sh; pkill -f train_uiclip.py
#
# Disk: the UIClip frozen cache already exists; CLIP and the fine-tuning run
# (layer-8 cache) each build a new ~53 GB cache on first use.

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
mkdir -p outputs/logs

EPOCHS=10
SEEDS=(0 1 2)

run() {  # usage: run <experiment> <seed> [extra train_uiclip.py flags...]
  local exp="$1_${EPOCHS}_epochs_s$2" seed=$2
  shift 2
  echo "$(date '+%F %T') start  $exp"
  if python scripts/train_uiclip.py --exp "$exp" --seed "$seed" --epochs "$EPOCHS" --cache-features "$@" \
       > "outputs/logs/$exp.log" 2>&1; then
    echo "$(date '+%F %T') done   $exp"
  else
    echo "$(date '+%F %T') FAILED $exp (see outputs/logs/$exp.log)"
  fi
}

# Seeds on the outside: one result per experiment arrives early.
for seed in "${SEEDS[@]}"; do
  run uiclip_cal          "$seed"                                              # main model
  run uiclip_nocal        "$seed" --no-self-calibration                        # identical-pair fix off
  run clip_cal            "$seed" --checkpoint openai/clip-vit-base-patch32    # UI pretraining vs plain CLIP
  run uiclip_cal_nobenign "$seed" --exclude-benign                             # benign training pairs removed
  run uiclip_cal_ft4      "$seed" --train-last-n 4                             # fine-tune last 4 encoder blocks
done

echo "$(date '+%F %T') scoring"
python scripts/score_predictions.py outputs/*_"${EPOCHS}"_epochs_s*/best_auroc/val_predictions.jsonl \
  --out outputs/comparison
echo "$(date '+%F %T') all done; see outputs/comparison/comparison.md"
