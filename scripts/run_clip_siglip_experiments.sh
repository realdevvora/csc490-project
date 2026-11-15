#!/usr/bin/env bash
# Run the follow-up experiments: CLIP with its last 4 blocks fine-tuned, and SigLIP.
#
#   siglip_cal_10_epochs_s{0,1,2}      SigLIP-512, frozen, cached       (does a finer patch grid help?)
#   clip_cal_ft4_10_epochs_s{0,1,2}    CLIP, last 4 blocks fine-tuned   (does UIClip's edge survive fine-tuning?)
#   siglip_cal_ft4_10_epochs_s0        SigLIP-512, last 4 blocks, uncached, one seed
#
# The first SigLIP run builds its feature cache (~285 GB) and the first CLIP
# fine-tuning run builds another (~49 GB). SigLIP fine-tuning runs without a
# cache: a second 285 GB cache would barely speed it up. Runs go to
# outputs/<experiment>/ and logs to outputs/logs/; a failed run is reported and
# the rest continue. At the end every 10-epoch run is rescored into
# outputs/comparison/.
#
# Start it in the background (keeps running if the terminal closes):
#   mkdir -p outputs/logs && nohup bash scripts/run_clip_siglip_experiments.sh > outputs/logs/clip_siglip.log 2>&1 &
# Watch progress:     tail -f outputs/logs/clip_siglip.log
# Watch one run:      tail -f outputs/logs/siglip_cal_10_epochs_s0.log
# Stop everything:    pkill -f run_clip_siglip_experiments.sh; pkill -f train_uiclip.py

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
mkdir -p outputs/logs

EPOCHS=10
CLIP=openai/clip-vit-base-patch32
SIGLIP=google/siglip-base-patch16-512
SIGLIP_CACHE=data/processed/features/google__siglip-base-patch16-512/layer12_w1536

run() {  # usage: run <experiment> <seed> [extra train_uiclip.py flags...]
  local exp="$1_${EPOCHS}_epochs_s$2" seed=$2
  shift 2
  echo "$(date '+%F %T') start  $exp"
  if python scripts/train_uiclip.py --exp "$exp" --seed "$seed" --epochs "$EPOCHS" "$@" \
       > "outputs/logs/$exp.log" 2>&1; then
    echo "$(date '+%F %T') done   $exp"
  else
    echo "$(date '+%F %T') FAILED $exp (see outputs/logs/$exp.log)"
  fi
}

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

# SigLIP frozen first, so a problem with the new backbone shows up early.
# Skip it if its cache isn't built yet and there isn't room for it.
if [ -d "$SIGLIP_CACHE" ] || [ "$(free_gb)" -ge 320 ]; then
  for seed in 0 1 2; do
    run siglip_cal "$seed" --checkpoint "$SIGLIP" --cache-features
  done
else
  echo "$(date '+%F %T') SKIPPED siglip_cal: needs ~285 GB for its cache, only $(free_gb) GB free"
fi

for seed in 0 1 2; do
  run clip_cal_ft4 "$seed" --checkpoint "$CLIP" --train-last-n 4 --cache-features
done

run siglip_cal_ft4 0 --checkpoint "$SIGLIP" --train-last-n 4

echo "$(date '+%F %T') scoring"
python scripts/score_predictions.py outputs/*_"${EPOCHS}"_epochs_s*/best_auroc/val_predictions.jsonl \
  --out outputs/comparison
echo "$(date '+%F %T') all done; see outputs/comparison/comparison.md"
