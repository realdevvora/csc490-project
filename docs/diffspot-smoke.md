# Model-agnostic 30-50 pair DiffSpot walkthrough

Open `notebooks/diffspot_smoke.ipynb` for guided cells. The same sample,
runner, scoring and visual review work for all group members, independently
of architecture or ML framework. No model is trained by this workflow.

Default: 44 pairs = 39 changed (one per native operator/difficulty cell) plus
5 no-diff controls, seed 490. Save once and reuse across models. `--n` accepts
30-50. Sampling uses annotations only for stratification; prediction receives
ordered images only. No GT crop, operator or sample ID enters the predictor.

These are preliminary operational checks, not validation/model-selection data.
Choose weights, thresholds, preprocessing and prompts on Design2Code before
looking at results. Performance-informed changes must be disclosed as an
exploratory round under ADR 0007. Do not tune against this small sample.

## 1. Save a shared sample

Use each member's existing model environment with the project's `evaluation`
dependencies and Pillow. Lab commands always use python3.

```powershell
.\.venv\Scripts\python.exe scripts/diffspot_smoke.py sample `
  --dataset D:\csc409-data\diffspot-v1 --n 44 `
  --out outputs/diffspot-smoke/sample44
```

No screenshots are copied. Share `sample.json` and `inference.jsonl` together.
All members use the exact same prepared dataset manifest, regardless of where
it was extracted. Keep local output/review files ignored.

## 2. Run ANY model (or import its existing predictions)

Option A: use your existing inference runner to write the selected sample IDs
in the shared JSONL format. Go directly to review. There is no need to rewrite
an already working predictor.

Option B: provide a small Python adapter in your own model code:

```python
# my_model_adapter.py (importable from your working environment)
def create_predictor(config):
    model = load_my_existing_model(config)  # your checkpoint/config/device
    model.eval()  # or your framework's inference equivalent

    def predict_pair(expected_image, actual_image):
        # Inputs are PIL RGB images at original resolution, expected first.
        # Reuse your trained preprocessing, inference and box post-processing.
        result = run_my_existing_inference(model, expected_image, actual_image)
        return result
    return predict_pair
```

The two model-specific function names above are deliberately pseudocode, not
provided model implementations. Each owner supplies their existing model's
bridge. There is no DINO-specific runner or fake/random model fallback.
The factory is loaded once. Prediction returns a Python dict containing a
finite `score`, and optionally `box`, `point`, `op_pred`. Coordinates must be
mapped back to original pixels. Missing localization/type outputs are null
or omitted; type names use the shared Design2Code taxonomy. Return ordinary
Python floats/lists rather than tensors. The harness attaches the sample ID.

The config is an owner-defined JSON object passed unchanged to the factory;
that code owns checkpoint loading, device selection, seeds, eval mode and
preprocessing. Resolve checkpoint paths explicitly. The harness makes no
assumptions about checkpoint serialization or framework. The adapter module
must be importable (installed package or configured PYTHONPATH).

```bash
python3 scripts/diffspot_smoke.py predict   --dataset /path/to/diffspot-v1   --sample outputs/diffspot-smoke/sample44   --adapter my_model_adapter:create_predictor   --model-config /path/to/my-selected-model.json   --out outputs/diffspot-smoke/my-model
```

Outputs: predictions.jsonl, run.json and complete.json. Sample/config/runner
hashes are recorded. Failed runs are not treated as successful predictions;
use a new output namespace when retrying.

## 3. Score and inspect

```bash
python3 scripts/diffspot_smoke.py review   --dataset /path/to/diffspot-v1   --sample outputs/diffspot-smoke/sample44   --predictions outputs/diffspot-smoke/my-model/predictions.jsonl   --validation-metrics /path/to/my-design2code-validation.metrics.json   --out outputs/diffspot-smoke/my-review
```

Use the actual Design2Code shared-scorer validation file (1,033 pairs); its
threshold is reused without optimization. Add `--supports-type` only for a
model with a type head. `--from-full` explicitly selects the sample from a
complete full-DiffSpot prediction file; missing/duplicate IDs are rejected.

Outputs: preliminary metrics.json, per-case cases.csv and self-contained
review.html. Green boxes show the visible target element; blue boxes show
predictions. The HTML embeds images and can be copied from lab to local.
Review annotations are post-inference only, never model/VLM inputs. Missing
model capabilities remain unavailable; no type/box prediction is fabricated.
Official description scoring requires the separate full
[DiffSpot workflow](diffspot-benchmark.md). No Kimi/judge calls are made here.
