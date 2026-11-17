# Heatmap-derived box IoU: post-hoc validation extension

Added 2026-11-16. Original A/B/C v1 checkpoints, scores, points, splits and
reports stay intact. This adds a deterministic spatial post-processing stage;
it does not train a new model or add mutation-type predictions. The lab maps have now been processed; see the verified results in the
[main report](trained-validation-results.md#verified-box-localization-extension-2026-11-16).

Decision record: [ADR 0006](../../adr/0006-posthoc-dino-box-localization.md).

## Fixed procedure

Rule `half_peak_largest_4connected_v1`: for each nonnegative heatmap, mask
invalid/padded cells and threshold at **half its valid maximum**. Box the
largest 4-connected foreground component; ties use row-major component order.
For an exactly zero map, propose the first valid peak cell instead. This is
a localization fallback, not an assertion that a regression exists.

Use the saved 2-pixel grid, scale to screenshot coordinates and clip the box
to the union image dimensions (including odd edges). Keep original detection
scores and points. Boxes are exported for every pair without consulting its
label or mutation location; the shared scorer evaluates IoU only on true
regressions against `diff_bbox`. For reflow, pointing still uses `bbox_after`,
so pointing and IoU assess different targets. No segmentation metric is claimed.

This rule was fixed before inspecting its IoU results. It does not share the
screenshot detection threshold and is not a calibrated probability cutoff.
The same relative rule applies to A's cosine maps and B/C's learned location
maps. No validation threshold search or IoU-based checkpoint selection is run.
The result is a **post-hoc validation extension**, not independent evaluation.
Diffuse activations can produce oversized boxes; record that failure honestly.

## Run on the lab (CPU only)

First transfer these new source files to the lab checkout using your usual
working-tree transfer. Use the existing environment; no new package is needed
beyond the project's evaluation dependencies (NumPy and SciPy).

```bash
export LAB_ROOT="/virtual/.cache/$USER/csc490"
cd "$LAB_ROOT/csc490-project"

# Use the ORIGINAL completed evaluation directories, not the map-free export.
# If your v1 results live elsewhere, change HEAD_ROOT to that exact directory.
HEAD_ROOT=outputs/dino_heads

"$LAB_ROOT/.venv/bin/python3" scripts/export_dino_boxes.py \
  --evaluation outputs/fullpage/dinov2_val outputs/fullpage/dinov3_val \
    "$HEAD_ROOT/dinov2_validation" "$HEAD_ROOT/dinov3_validation" \
  --data-root "$LAB_ROOT/data/d2c_v3" \
  --out outputs/dino_boxes_v1

cat outputs/dino_boxes_v1/comparison.md
```

Alternatively `bash scripts/run_dino_box_export.sh` runs the command and
packages its output as `$LAB_ROOT/dino-boxes-v1.tar.gz`. Override `HEAD_ROOT`
only when the original evaluation directories have a different location;
the default deliberately uses `outputs/dino_heads`, not corrected v2 runs.

An A evaluation needs `config.json`, `complete.json`, `predictions.jsonl` and
`maps/00000.npz` onwards (`cosine`, `valid`). A head evaluation needs its config,
completion record, `selection.json`, and selected checkpoint subdirectories
containing predictions and numbered maps (`location`, `valid`). Map identity
comes from the source config's `sample_ids`, not the prediction row order.

The command preflights dataset/split hashes, full validation coverage and all
required maps. A missing-map error means use the original lab directories or
restore maps; do not retrain. If maps were deleted, inference is required.
Head processing reuses the six already selected runs per backbone (B/C × three
seeds), not all intermediate checkpoints. Do not mix corrected v2 maps with v1
predictions. A different output namespace is required for corrected runs.

Outputs: one augmented prediction file, shared metrics and map/box provenance
per run; `summary.json` and `comparison.md` aggregate per-seed IoU with sample
SD; `complete.json` marks successful completion. The exporter asserts that
all non-localization metrics and original pointing metrics remain identical.
It refuses nonempty destinations. Source artifacts are never overwritten.

To copy the small results back, archive the exporter output alone:

```bash
tar -czf "$LAB_ROOT/dino-boxes-v1.tar.gz" outputs/dino_boxes_v1
```

Inspect/import those results before adding numerical IoU to the main report
or poster handoff. Do not replace the original v1 report's missing metrics
with estimates. Mutation-type macro-F1 remains unavailable.
