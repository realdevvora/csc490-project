#!/usr/bin/env bash
# Rerun every UIClip/CLIP/SigLIP experiment on the team split, in one batch.
#
# Both earlier batches (run_uiclip_experiments.sh, run_clip_siglip_experiments.sh)
# in one script, trained on data/splits/splits_85_15.json, which leaves out the
# pages with a broken no_change pair (ADR 0004). Runs go to
# outputs/team_split/<experiment>_10_epochs_s<seed>/, apart from the earlier
# runs, and logs to outputs/logs/team_split/. At the end every run is scored
# into outputs/team_split/comparison/.
#
#   experiment            backbone   encoder          seeds
#   uiclip_cal            UIClip     frozen           0 1 2   main model
#   uiclip_nocal          UIClip     frozen           0 1 2   identical-pair fix off
#   uiclip_cal_nobenign   UIClip     frozen           0 1 2   no benign pairs in training
#   clip_cal              CLIP       frozen           0 1 2   UI pretraining vs plain CLIP
#   siglip_cal            SigLIP-512 frozen           0 1 2   finer patch grid
#   uiclip_cal_ft4        UIClip     last 4 tuned     0 1 2
#   clip_cal_ft4          CLIP       last 4 tuned     0 1 2
#   siglip_cal_ft4        SigLIP-512 last 4 tuned     0       uncached, ~3.4 h per seed
#
# Resumable: a run whose metrics.json already has all its epochs is skipped, so
# after a crash or reboot just start the script again. An interrupted run
# restarts from epoch 1. A failed run is reported and the rest continue.
#
# Options (environment variables):
#   EXPERIMENTS="uiclip_cal clip_cal"   run only these (default: all of the above)
#   SEEDS="0 1 2"                       seeds for every experiment except siglip_cal_ft4
#   SIGLIP_FT4_SEEDS="0"                seeds for siglip_cal_ft4 ("0 1 2" adds ~7 h)
#   EPOCHS=10
#   ALLOW_DIRTY=1                       run even with uncommitted changes (recorded as dirty)
#
# Start it in the background (keeps running if the terminal closes):
#   mkdir -p outputs/logs/team_split && nohup bash scripts/run_all_experiments.sh > outputs/logs/team_split/all.log 2>&1 &
# Watch progress:     tail -f outputs/logs/team_split/all.log
# Watch one run:      tail -f outputs/logs/team_split/uiclip_cal_10_epochs_s0.log
# Stop everything:    pkill -f run_all_experiments.sh; pkill -f train_uiclip.py

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate || exit 1

EPOCHS=${EPOCHS:-10}
SEEDS=${SEEDS:-"0 1 2"}
SIGLIP_FT4_SEEDS=${SIGLIP_FT4_SEEDS:-"0"}
ALL="uiclip_cal uiclip_nocal uiclip_cal_nobenign clip_cal siglip_cal uiclip_cal_ft4 clip_cal_ft4 siglip_cal_ft4"
EXPERIMENTS=${EXPERIMENTS:-$ALL}
GROUP=team_split
SPLIT=data/splits/splits_85_15.json
LOGS=outputs/logs/$GROUP
CLIP=openai/clip-vit-base-patch32
SIGLIP=google/siglip-base-patch16-512
SIGLIP_CACHE=data/processed/features/google__siglip-base-patch16-512/layer12_w1536
mkdir -p "$LOGS"

log() { echo "$(date '+%F %T') $*"; }
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

# --- Checks before starting a run that may take ~11 hours ---------------------
for name in $EXPERIMENTS; do
  [[ " $ALL " == *" $name "* ]] || { log "unknown experiment '$name' (known: $ALL)"; exit 1; }
done
[ -f d2c_v2/master_dataset_with_benign.json ] || { log "dataset missing: d2c_v2/master_dataset_with_benign.json"; exit 1; }
python - "$SPLIT" <<'EOF' || exit 1
import json, sys
from regression_lens.data.records import load_records
from regression_lens.data.splits import load_split, make_team_split
path = sys.argv[1]
split = json.load(open(path))
if "excluded" not in split:
    sys.exit(f"{path} has no 'excluded' list: this is the old team split; update from main first")
records = load_records("d2c_v2")
load_split(path, records)
if make_team_split(records) != split:
    sys.exit(f"{path} doesn't match the dataset in d2c_v2/ (different dataset version?)")
print(f"split ok: {len(split['train'])} train / {len(split['val'])} val pages, {len(split['excluded'])} excluded")
EOF
if [ -n "$(git status --porcelain)" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
  log "uncommitted changes; every run's config.json would record dirty=true. Commit or stash them,"
  log "or set ALLOW_DIRTY=1. Changes:"; git status --short; exit 1
fi
nvidia-smi > /dev/null 2>&1 || { log "no GPU visible (nvidia-smi failed)"; exit 1; }
log "commit $(git rev-parse --short HEAD), $(free_gb) GB free, experiments: $EXPERIMENTS"

# --- Runs ---------------------------------------------------------------------
wanted() { [[ " $EXPERIMENTS " == *" $1 "* ]]; }

finished() {  # true if outputs/<exp>/metrics.json has all $EPOCHS epochs
  python -c "import json,sys; sys.exit(len(json.load(open(sys.argv[1]))['history']) < int(sys.argv[2]))" \
    "outputs/$1/metrics.json" "$EPOCHS" 2>/dev/null
}

run() {  # usage: run <experiment> <seed> [extra train_uiclip.py flags...]
  local name=$1 seed=$2
  local exp="$GROUP/${name}_${EPOCHS}_epochs_s${seed}"
  shift 2
  wanted "$name" || return 0
  if finished "$exp"; then log "skip   $exp (already finished)"; return 0; fi
  log "start  $exp"
  if python scripts/train_uiclip.py --exp "$exp" --seed "$seed" --epochs "$EPOCHS" --split "$SPLIT" "$@" \
       > "$LOGS/${name}_${EPOCHS}_epochs_s${seed}.log" 2>&1; then
    log "done   $exp"
  else
    log "FAILED $exp (see $LOGS/${name}_${EPOCHS}_epochs_s${seed}.log)"
  fi
}

siglip_cache_ok=1
if wanted siglip_cal && [ ! -d "$SIGLIP_CACHE" ] && [ "$(free_gb)" -lt 320 ]; then
  log "SKIPPING siglip_cal: its cache (~285 GB) isn't built and only $(free_gb) GB are free"
  siglip_cache_ok=0
fi

# Seeds on the outside, so every experiment has one result early; cheap frozen
# runs before fine-tuned ones in each round.
for seed in $SEEDS; do
  run uiclip_cal          "$seed" --cache-features
  run uiclip_nocal        "$seed" --cache-features --no-self-calibration
  run uiclip_cal_nobenign "$seed" --cache-features --exclude-benign
  run clip_cal            "$seed" --cache-features --checkpoint "$CLIP"
  [ $siglip_cache_ok = 1 ] && run siglip_cal "$seed" --cache-features --checkpoint "$SIGLIP"
  run uiclip_cal_ft4      "$seed" --cache-features --train-last-n 4
  run clip_cal_ft4        "$seed" --cache-features --train-last-n 4 --checkpoint "$CLIP"
done
# Longest and least important last: no cache, ~3.4 h per seed.
for seed in $SIGLIP_FT4_SEEDS; do
  run siglip_cal_ft4 "$seed" --train-last-n 4 --checkpoint "$SIGLIP"
done

# --- Scoring --------------------------------------------------------------------
shopt -s nullglob
predictions=(outputs/"$GROUP"/*_"${EPOCHS}"_epochs_s*/best_auroc/val_predictions.jsonl)
if [ ${#predictions[@]} -eq 0 ]; then log "no finished runs to score"; exit 1; fi
log "scoring ${#predictions[@]} runs"
python scripts/score_predictions.py "${predictions[@]}" --split "$SPLIT" --out "outputs/$GROUP/comparison"
log "all done; see outputs/$GROUP/comparison/comparison.md"
