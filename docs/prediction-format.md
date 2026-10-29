# Prediction format and shared scoring

Any model, whatever its architecture, can be compared with the others if it writes predictions in this format. `scripts/score_predictions.py` scores those files with the shared metrics, and the code is in `src/regression_lens/evaluation/scoring.py`.

Format version: **1**

## File

- [JSON Lines](https://jsonlines.org/): one JSON object per line, UTF-8, one line per screenshot pair. Blank lines are ignored.
- One file per model and evaluated subset, e.g. `uiclip_frozen_val.jsonl` or `dinov3_test.jsonl`.
- The file must contain **exactly one prediction for every pair in the evaluated subset**. The subset is every pair whose `parent_id` is in the split's `val` (or `test`) page list. That includes all regression, `no_change` and `benign` pairs. Missing, duplicate or extra `sample_id`s are errors.
- Predict every pair without looking at its label. Ground truth comes only from the dataset metadata (`master_dataset_with_benign.json`), never from the prediction file.

## Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `sample_id` | string | yes | The pair's `sample_id` from the metadata, e.g. `"d2c_000_font_size"`. |
| `score` | number | yes | Regression confidence; higher means more likely a regression. A probability in [0, 1] is recommended, but any finite score with a consistent direction works. |
| `op_pred` | string or null | no | Predicted mutation operator; one of the 13 names below. Scored on true regressions only. |
| `box` | `[x, y, w, h]` or null | no | Main predicted changed region. Scored with IoU against `diff_bbox`. |
| `point` | `[x, y]` or null | no | Single most-likely changed location. Scored with pointing accuracy. If missing, the centre of `box` is used. |

Any other fields, such as `model` or `format_version`, are allowed and ignored. Omit an optional field or set it to `null` if your model doesn't produce it. The scorer then reports that metric as `-` and records how many pairs were covered.

**Operators** (`op_pred` values): `background_color`, `border_radius`, `border_width`, `font_size`, `font_weight`, `letter_spacing`, `line_height`, `margin`, `opacity`, `padding`, `position`, `text_color`, `width`.

**Coordinates** use pixels of the **original screenshot**, not your model's resized input. The origin is the top-left corner, x goes right and y goes down, the same as `diff_bbox` and `bbox_after` in the metadata. If your model works on a resized image, divide by the scale factor before writing. For example, if 1280 px became 1344 px, divide by 1.05. `w` and `h` must be positive.

**Masks and heatmaps** aren't stored in this file. Turn them into a box and a point first:
- **box:** threshold the mask and box the largest connected region, e.g. with `regression_lens.evaluation.metrics.largest_component_box`.
- **point:** the location of the highest value.

## Example

```json
{"sample_id": "d2c_000_identical", "score": 0.03, "op_pred": "opacity", "box": [610.0, 1402.3, 30.5, 30.5], "point": [625.2, 1417.5]}
{"sample_id": "d2c_000_background_color", "score": 0.97, "op_pred": "background_color", "box": [30.5, 1493.3, 579.0, 335.2], "point": [289.5, 1615.2]}
{"sample_id": "d2c_000_firefox", "score": 0.41, "op_pred": null, "box": null, "point": null}
```

The scorer only evaluates `op_pred`, `box` and `point` on true regressions, so values given for benign or no-change pairs don't affect any metric.

## Metrics

| Group | Metric | Notes |
|---|---|---|
| Threshold-free detection | AUROC, regression vs. benign | Main comparison number. Excludes `no_change`, which is pixel-identical and trivially separable. |
| | AUROC, regression vs. all negatives | Includes `no_change`. |
| | Benign FPR at 95% recall | The threshold that catches 95% of regressions, and the share of benign pairs it flags. |
| Detection at a threshold | Precision, recall, F1, accuracy | Regression is the positive class. Precision and F1 depend on the share of regressions in the data (about 80% here). |
| | FPR on `benign` and on `no_change` | Reported separately. |
| Change category | Type macro-F1, per-class F1 and support | Over true regressions that have an `op_pred`. |
| Localization | Pointing accuracy | Is `point` inside `diff_bbox`? For reflowed mutations, is it inside `bbox_after`, because their `diff_bbox` covers everything that moved? |
| | Mean box IoU | Between `box` and `diff_bbox`. |
| Breakdowns | Recall by family, magnitude and reflow | At the chosen threshold. |

## Choosing the threshold

The threshold-based metrics depend on how the threshold is chosen, so the scorer records the method in `threshold_selection`:

- **Scoring validation predictions:** by default, the scorer uses the threshold that maximizes F1 on the scored file.
- **Scoring test or external predictions:** pass `--threshold-from <your val predictions>`, which takes the max-F1 threshold from validation, or pass a fixed `--threshold`. Choosing the threshold on test data overstates results.

Threshold-free metrics (AUROC, FPR at 95% recall) don't depend on this choice. That's why they are the fairest basis for comparing models.

## Running the scorer

```bash
# One model's validation predictions
python scripts/score_predictions.py outputs/uiclip_frozen/best_auroc/val_predictions.jsonl

# Compare several models; writes <name>.metrics.json and comparison.md
python scripts/score_predictions.py preds/uiclip_val.jsonl preds/dinov3_val.jsonl preds/bit_val.jsonl \
    --names uiclip dinov3 bit --out outputs/comparison

# Test predictions, each with the threshold from its own validation predictions
python scripts/score_predictions.py preds/uiclip_test.jsonl preds/dinov3_test.jsonl --subset test \
    --threshold-from preds/uiclip_val.jsonl preds/dinov3_val.jsonl
```

`--split` (default: `data/processed/splits_placeholder_85_15.json`) and `--subset` (default: `val`) define which pairs must be covered. `--subset all` requires every pair in the dataset. Everyone in the group must use the same split file.

## Not covered yet

- **DiffSpot:** it needs an adapter that maps its records and labels to these metrics, plus Easy/Medium/Hard breakdowns.
- **Pixel IoU** for mask-producing models.
- **Bootstrap confidence intervals.**
