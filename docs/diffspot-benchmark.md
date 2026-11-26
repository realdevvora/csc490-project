# DiffSpot: data preparation and frozen Test 1 / Test 2 scoring

Implemented on `abu-diffspot-data`, 2026-11-25. This is a dataset/scoring
foundation, not completed model benchmarking. [ADR 0007](adr/0007-frozen-full-diffspot-benchmark.md)
supersedes the former half-split protocol. Use all 4,400 official pairs; freeze
all four configurations and shared Kimi settings on Design2Code first. No
DiffSpot training or tuning is permitted.

## Setup and prepare

For an architecture-independent 30-50 pair walkthrough, use
[the smoke-test guide](diffspot-smoke.md) and
[guided notebook](../notebooks/diffspot_smoke.ipynb). It reuses one shared
sample across model owners, accepts existing predictions or model adapters,
and does not substitute smoke results for the full benchmark.

Use a separate data environment so dependency updates do not disturb model
environments. Lab commands always use python3; on Windows use the equivalent
environment interpreter. No GPU, torch, model weights or API key is required.

```bash
python3 -m venv .venv-diffspot
source .venv-diffspot/bin/activate
python3 -m pip install -e ".[diffspot]"

# Small online smoke: never report as full-benchmark results.
python3 scripts/diffspot.py prepare --limit-pairs 4 \
  --out data/external/diffspot-smoke

# Full preparation; the audited immutable HF revision is also the default.
python3 scripts/diffspot.py prepare --revision c6dd79d5e1c0cbb4e7ca234c9f53c418a75e30ce \
  --out data/external/diffspot-full
```

Keep `.venv-diffspot/` outside Git. A local import is also supported:
`prepare --source-jsonl PATH --out DIRECTORY`, using native release column
names and local `image_before`/`image_after` paths relative to the JSONL file.
Its source JSONL checksum becomes the revision. Use `--limit-pairs` for fixtures.

Outputs include `images/`, `inference.jsonl`, `annotations.jsonl`,
`mapping-coverage.json` and `manifest.json`. The release has 3,900 changed pairs
(1,300 per tier, 300 per operator) and 500 no-diff controls. The official loader also
supports historical revisions with 28 `unknown`-difficulty controls; we match
its no_diff routing. At the audited revision all 500 controls already use
no_diff. No records are silently discarded.

On a machine whose HTTPS proxy/root certificates live in the OS store, add
`prepare --system-trust`. This retains TLS verification. Use an up-to-date pip
for installation as well; do not disable certificate checks.

Give inference code **only inference.jsonl and its images**. It contains opaque
IDs/paths, image sizes and hashes. Annotations contain native mutation details,
descriptions, tiers and target boxes for scoring only. Do not pass annotations,
operator-derived filenames or judge requests to Kimi. The manifests/checksums
are reproducibility boundaries, not access-control isolation.

## Freeze the selected configurations

Copy `configs/diffspot.example.json` to a local ignored location and replace
every placeholder. Relative paths are resolved from the specification's folder.
Each owner supplies a real checkpoint SHA256, seed policy, preprocessing, box
rule, type-head capability and **shared Design2Code validation metrics for all
1,033 pairs**. The threshold is read from those metrics; no DiffSpot optimization
is available. The data tool records owner-supplied weight fingerprints rather
than loading weights. Preserve the four owners' selection evidence separately.

```bash
python3 scripts/diffspot.py freeze --dataset data/external/diffspot-full \
  --spec outputs/diffspot/selected-configurations.json \
  --out outputs/diffspot/frozen.json
```

All four pipelines use one Kimi K2.5 identity/prompt: one proposed region per
pair, aligned crops with 32px padding, 1280px-wide overview, expected then actual,
temperature 0 and 16,384 output-token budget. These are the frozen input/output
contract; this delivery does not implement the API runner. The same prompt is
used regardless of ground-truth task/tier. Set the prompt file path correctly
after copying the example. The committed shared prompt defines verification
only: accept/reject/unsure plus a visible-effect description, no box coordinates.

Fresh output paths are required. A smoke manifest requires explicit
`--allow-preliminary` and all resulting scores remain preliminary. Changes to
scorer code, dataset or frozen settings require a separately declared round.

## Test 1: detector predictions

Each model owner runs their frozen model over the inference manifest and writes
the shared base JSONL shape:

```json
{"sample_id":"OPAQUE_SOURCE_ID","score":0.8,"op_pred":null,"box":[20,30,40,50],"point":[40,55]}
```

Coordinates are original screenshot pixels. Base type labels use Design2Code's
taxonomy. Every prepared ID must appear exactly once. Do not rescale coordinates
twice, use a ground-truth box or drop new operators. The dataset module does
not create fake Design2Code metadata; model owners adapt their image loaders to
the inference manifest. This avoids implicitly injecting mutation locations.

```bash
python3 scripts/diffspot.py score-base --dataset data/external/diffspot-full \
  --frozen outputs/diffspot/frozen.json --run selected-dino \
  --predictions outputs/diffspot/dino-base.jsonl --out outputs/diffspot/test1-dino
```

All six official report fields exist. Without real descriptive predictions and
official judge labels, description-dependent values are **null**, with explicit
availability status. `supplementary.detector_only_report` supplies the same
six tier/overall/no-diff slices using binary detection correctness; those are
detector metrics, not the paper's mutation-identification metrics. AUROC,
precision/recall/F1, fixed-threshold specificity, localization and type coverage
are additionally recorded. Diagnostic FPR@95R is a curve statistic, not a new
deployable threshold.

If a base model genuinely produces descriptions, submit its six-field official
file with `model` equal to the run ID and `prompt_version` equal to the frozen
`base_prompt_version` (default `base-v1`). Supply `--official-predictions`,
`--judged` and `--judge-metadata` together to enable official scoring. Never
generate its answer from ground-truth operator/description metadata.

## Test 2: exact official output plus a separate verification sidecar

The teammate's Kimi runner must emit exactly the official fields per pair:

```json
{"id":"OPAQUE_SOURCE_ID","split":"easy","model":"selected-dino+kimi-k2.5","prompt_version":"rl-verify-v1","raw_response":"Actual raw model response","prediction":"The affected button has more rounded corners."}
```

`model` is `<run ID>+<frozen Kimi model>`. The inference runner does not see
`split`; evaluation packaging joins it from annotations **after** inference.
Use `easy`, `medium`, `hard` or `no_diff`, including normalized controls.
Raw response and prediction must be nonempty strings. Timeout/error responses
are incomplete runs, never invented no-diff answers.

Separately emit one verification row per pair:

```json
{"sample_id":"OPAQUE_SOURCE_ID","decision":"accept","box":[20,30,40,50],"description":"The affected button has more rounded corners.","source_prediction_sha256":"SHA256_OF_ENTIRE_BASE_JSONL_FILE"}
```

Accepted boxes must equal the corresponding base box exactly. Rejected/unsure
boxes are null and the final description is exactly `No differences identified.`
Their raw Kimi responses remain in the official file. Record unsure separately.
The detector may propose a box even below its binary detection threshold;
Kimi can verify it. No accepted proposal can invent a missing detector box.

### Official judging without API calls in this CLI

```bash
python3 scripts/diffspot.py judge-inputs --test 2 \
  --dataset data/external/diffspot-full --frozen outputs/diffspot/frozen.json \
  --run selected-dino --predictions outputs/diffspot/dino-kimi.jsonl \
  --out outputs/diffspot/dino-judge-inputs
```

This produces **evaluator-only** `judge_requests.jsonl` (with GT) and
`judge-metadata.json`. Send these jobs to a separately authorized judge runner,
not to Kimi inference. Use pinned official `judge_item` with `gpt-oss-120b`,
v2.0 prompt, high reasoning effort, temperature 0, max_tokens 16384. Its API
access, cost and execution remain a later integration step. Kimi is the system
being evaluated, not the default judge.

Each returned judge row matches the official CLI's internal judged-item shape:

```json
{"id":"OPAQUE_SOURCE_ID","split":"easy","operator":"rounded","judge_label":{"mutations":[{"gt":"judge's GT reference","type":"rounded","verdict":"correct","vlm_match":"matched response"}],"hallucinations":[],"summary":{"correct":1,"partial":0,"missed":0,"hallucinated":0}},"is_true_positive":true,"is_true_negative":false,"n_correct_mutations":1,"n_hallucinations":0}
```

The last four fields are optional redundant reducer outputs; if supplied, they
must agree with the official reducer. All IDs must be returned exactly once.
Keep the generated judge metadata with the judgments as provenance. It binds
prediction bytes, dataset/frozen hashes, prompt and judge settings; metadata
does not independently prove which remote model was called. The external
runner must record real calls/settings honestly. Aggregate-only official
score files cannot substitute for per-case labels for audit/pairing purposes.

```bash
python3 scripts/diffspot.py score-vlm --dataset data/external/diffspot-full \
  --frozen outputs/diffspot/frozen.json --run selected-dino \
  --base-predictions outputs/diffspot/dino-base.jsonl \
  --predictions outputs/diffspot/dino-kimi.jsonl \
  --decisions outputs/diffspot/dino-verification.jsonl \
  --judged outputs/diffspot/dino-judged.jsonl \
  --judge-metadata outputs/diffspot/dino-judge-inputs/judge-metadata.json \
  --out outputs/diffspot/test2-dino
```

`official_predictions.jsonl` has exactly the six official prediction fields.
`official_scores.json` has the official CLI envelope: judge_version, judge_model,
reasoning_effort, n_predictions, n_judged, report. The report is produced by
the unmodified official aggregator/reducer: Easy/Medium/Hard Recall, Diff Overall
Recall, No-Diff Specificity, Overall Accuracy, counts and per-operator recall.
Only `correct` counts as a changed-case TP; `partial` is not a TP. Supplementary
project metrics live separately in `metrics.json`, never in the official report.

## Localization, taxonomy and combined reporting

DiffSpot boxes identify the mutated element. Preserve `source_target_box`
exactly; `target_box` is its visible intersection with the screenshot canvas.
330 native boxes extend beyond the 1280x800 viewport in the audited metadata;
none are fully outside it. This clipping affects only our additional localization
metrics, never the native descriptions or official description scores. These
are not the same target as Design2Code's changed-pixel bbox. All changed cases remain in the localization
denominator. Missing/rejected boxes contribute zero to all-case IoU and grounded
recall; conditional IoU/pointing and support are also reported. Correctly
described-and-localized recall requires official identification and IoU ≥ .5.
Kimi cannot improve retained coordinates; filtering may change coverage and
conditional statistics. Do not call that geometric refinement.

See the aggregate [preparation audit](results/diffspot/preparation-audit.json).

The versioned mapping is `src/regression_lens/data/diffspot_taxonomy_v1.json`.
Exact aliases map known properties; color/spacing require explicit structured
properties. Border colour is not border width; gaps are not margin. Unmapped
types remain in both tests. Native properties include `text-shade`/`background-shade`, explicit padding
and explicit margin names. These are mapped by property; gap changes remain
unmapped. The audited release has 2,616 mapped changed pairs and 1,284 unmapped
changed pairs; all 3,900 remain in both tests. Only supplementary base type
macro-F1 uses the mapped subset. Missing type predictions count as errors for a declared type
head; models without that head report unavailable. Description judging uses
the native DiffSpot taxonomy without any Design2Code mapping.

```bash
python3 scripts/diffspot.py report outputs/diffspot/test1-* outputs/diffspot/test2-* \
  --frozen outputs/diffspot/frozen.json --out outputs/diffspot/comparison
```

Pass result directories explicitly if shell wildcard expansion includes other
files. JSON/CSV/Markdown reports retain official and detector-only metrics,
denominators, localization support and paired deltas. A complete four-pipeline
comparison requires all eight full-release results from the same frozen round;
Test 2 must reference exactly the compared Test 1 prediction bytes. Unavailable
base description scores remain unavailable; no artificial official delta is made.

Published Kimi-only paper scores may be listed as external context. We do not
rerun that baseline, and do not infer a matched guidance improvement from it.

## Sources and reproducibility

Vendored unmodified code/prompts and Tencent's MIT notice are under
`src/regression_lens/evaluation/_diffspot_official/`. `PROVENANCE.json` records
upstream commit `3108fcdd35618f2cfd56ec173fe15850e6395334` and individual hashes;
freeze/scoring verify them. No third-party API function is called by this CLI.
The data revision is pinned independently at preparation. Frozen checksums
protect against accidental changes; they are not a substitute for selection
discipline or image-hash verification in each model owner's inference runner.

- [Official evaluation](https://github.com/Tencent/DiffSpot/blob/3108fcdd35618f2cfd56ec173fe15850e6395334/docs/evaluation.md)
- [Official dataset](https://huggingface.co/datasets/tencent/DiffSpot)
- [Official metrics](https://github.com/Tencent/DiffSpot/blob/3108fcdd35618f2cfd56ec173fe15850e6395334/diffspot/metrics.py)
