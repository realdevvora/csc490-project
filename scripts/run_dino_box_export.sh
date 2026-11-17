#!/usr/bin/env bash
# CPU-only extension of completed v1 runs. No model loading or training.
set -euo pipefail
export LAB_ROOT="${LAB_ROOT:-/virtual/.cache/$USER/csc490}"
cd "$LAB_ROOT/csc490-project"
HEAD_ROOT="${HEAD_ROOT:-outputs/dino_heads}"
BOX_OUT="${BOX_OUT:-outputs/dino_boxes_v1}"
"$LAB_ROOT/.venv/bin/python3" scripts/export_dino_boxes.py \
  --evaluation outputs/fullpage/dinov2_val outputs/fullpage/dinov3_val \
    "$HEAD_ROOT/dinov2_validation" "$HEAD_ROOT/dinov3_validation" \
  --data-root "$LAB_ROOT/data/d2c_v3" --out "$BOX_OUT"
cat "$BOX_OUT/comparison.md"
tar -czf "$LAB_ROOT/dino-boxes-v1.tar.gz" "$BOX_OUT"
