#!/usr/bin/env bash
# Run only after experiment A has finished. Uses the existing lab environment.
set -euo pipefail
export LAB_ROOT="${LAB_ROOT:-/virtual/.cache/$USER/csc490}"
export HF_HOME="$LAB_ROOT/huggingface"
export TMPDIR="$LAB_ROOT/tmp"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$LAB_ROOT/csc490-project"
mkdir -p "$TMPDIR"
PYTHON3="$LAB_ROOT/.venv/bin/python3"
DATA_ROOT="$LAB_ROOT/data/d2c_v3"
for model in dinov2 dinov3; do
    if [[ ! -f "outputs/fullpage/${model}_val/complete.json" ]]; then
        echo "Wait for the existing ${model} full validation run to finish."
        exit 1
    fi
done
# A fresh namespace preserves legacy v1 crop manifests, caches and running jobs.
HEAD_ROOT=outputs/dino_heads_v2
MANIFEST="$HEAD_ROOT/training_windows.json"
"$PYTHON3" scripts/dino_heads.py manifest --data-root "$DATA_ROOT" --manifest "$MANIFEST"
for model in dinov2 dinov3; do
    CACHE="$HEAD_ROOT/${model}_cache"
    RUNS="$HEAD_ROOT/${model}_runs"
    if [[ ! -f "$CACHE/complete.json" ]]; then
        "$PYTHON3" scripts/dino_heads.py extract --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
            --model "$model" --out "$CACHE"
    fi
    for condition in B C; do
        for seed in 490 491 492; do
            OUT="$RUNS/$condition/seed$seed"
            if [[ ! -f "$OUT/complete.json" ]]; then
                "$PYTHON3" scripts/dino_heads.py train --cache "$CACHE" --condition "$condition" \
                    --seed "$seed" --out "$OUT"
            fi
        done
    done
    EVALUATION="$HEAD_ROOT/${model}_validation"
    if [[ ! -f "$EVALUATION/complete.json" ]]; then
        "$PYTHON3" scripts/dino_heads.py evaluate --data-root "$DATA_ROOT" --model "$model" \
            --runs "$RUNS" --out "$EVALUATION"
    fi
    "$PYTHON3" scripts/dino_heads.py report --evaluation "$EVALUATION"
    "$PYTHON3" scripts/review_dino_heads.py --runs "$RUNS" --evaluation "$EVALUATION" --data-root "$DATA_ROOT"
done
"$PYTHON3" scripts/score_predictions.py \
    outputs/fullpage/dinov2_val/predictions.jsonl \
    outputs/fullpage/dinov3_val/predictions.jsonl \
    outputs/fullpage/dinov2_val/pixel_predictions.jsonl \
    --names dinov2 dinov3 pixel --data-root "$DATA_ROOT" --subset val \
    --out outputs/fullpage/validation_metrics
"$PYTHON3" scripts/summarize_dino_experiments.py \
    --baselines outputs/fullpage/validation_metrics \
    --v2-heads "$HEAD_ROOT/dinov2_validation" \
    --v3-heads "$HEAD_ROOT/dinov3_validation" \
    --out "$HEAD_ROOT/final_comparison"
