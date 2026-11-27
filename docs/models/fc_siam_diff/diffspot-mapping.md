# FC-Siam-diff on DiffSpot: output-to-field mapping

How each FC-Siam-diff output becomes a field in the DiffSpot benchmark files, for Test 1 (detector alone) and Test 2 (detector + Kimi K2.5). The rules for both tests are in [ADR 0007](../../adr/0007-frozen-full-diffspot-benchmark.md) and the [DiffSpot workflow](../../diffspot-benchmark.md); this doc pins down FC-Siam-diff's side of them.

This is a spec for the code, not something sent to Kimi. Kimi only ever receives the frozen prompt and the four images in section 3.

## 1. Pipeline

```
inference.jsonl ──► FC-Siam-diff ──► base prediction ──────────────► Test 1 (score-base)
                                          │
                                          ▼ box only
                                     Kimi K2.5 ──► verification row ─┐
                                                └► official row ─────┴► Test 2 (score-vlm)
```

## 2. FC-Siam-diff output → base prediction (Test 1)

One line per pair in `fc-siam-base.jsonl`, for all 4,400 pairs. The fields follow the [shared prediction format](../../prediction-format.md).

**Inputs**
- `reference_path` is the **expected** screenshot (before, image A).
- `candidate_path` is the **actual** screenshot (after, image B).
- Both come from `inference.jsonl`. Swapping them reverses every answer.
- Annotations are never read at this stage.

**Fields**

| Field | Comes from | Rule |
|---|---|---|
| `sample_id` | `inference.jsonl` | Copied unchanged. Every ID appears exactly once. |
| `score` | `page_record`, `SCORE_RULE = 'diff_mean'` | Mean regression confidence (`1 − p[no_regression]`) over the pixels that differ. `0.0` when the two screenshots are identical. |
| `op_pred` | `OP_RULE = 'weighted'` | The type with the largest summed confidence over flagged pixels (confidence > `PIXEL_THRESHOLD` 0.5). Always one of our 13 type names, and present even when nothing is flagged. |
| `box` | `BOX_RULE = 'all'` | `[x, y, w, h]` around **all** flagged pixels, in original screenshot pixels. `null` when no pixel is flagged. |
| `point` | `POINT_RULE = 'peak'` | `[x, y]` of the most confident pixel. `null` when `box` is `null`. |

**Extra fields.** `op_second`, `op_family` and `model` may also be written. The scorer ignores them.

**Detection verdict.** A pair counts as a regression when `score ≥ 0.0816` (run 3's threshold from `fc_siam_diff_val.metrics.json`), and this is never re-tuned on DiffSpot. The box doesn't depend on this threshold. It depends only on the 0.5 pixel threshold, so a pair can have a box while scoring below 0.0816.

**Identical screenshots** (`GATE_WITH_DIFF`) always give `score 0.0`, `box null` and `point null`.

**Checks (from `validate_base`)**
- `score` is a finite number.
- `op_pred` is one of the 13 types. The run declares `supports_type: true`.
- `box` has `w, h > 0` and lies within the larger of the two screenshots.
- `point` lies inside the screenshot.

## 3. Base prediction → Kimi input (Test 2)

**When Kimi is called:** only when `box` is not `null`, whatever the `score`. A pair with no box is answered without calling Kimi (section 4).

**What Kimi receives**, in this order, each image preceded by its text label:

| # | Label | Image |
|---|---|---|
| 1 | `Image A overview (expected):` | Expected screenshot, scaled to 1280 px wide (DiffSpot is already 1280 × 800, so unchanged) |
| 2 | `Image B overview (actual):` | Actual screenshot, same scaling |
| 3 | `Image A crop (expected), proposed region:` | Expected screenshot cropped to the region |
| 4 | `Image B crop (actual), proposed region:` | Actual screenshot cropped to the same region |

**Region**
- `box` padded by 32 px on every side and clipped to the page.
- The same pixels are cut from both screenshots. Each is first padded with white at the bottom and right to the larger size, so the two crops line up.
- No box is drawn on the crops.

**Prompt:** `configs/prompts/kimi_diffspot_verification_v1.txt` (`rl-verify-v1`), sent as the system message.

**Never sent to Kimi:** `score`, `op_pred`, `point`, `sample_id`, file names, or anything from `annotations.jsonl`.

**Settings**

| Setting | Value |
|---|---|
| Model | `kimi-k2.5`, via OpenRouter as `moonshotai/kimi-k2.5`, pinned to Amazon Bedrock with no fallback |
| Temperature | 0 |
| Max output tokens | 16,384 |
| Thinking | Off. Bedrock's K2.5 doesn't think even when asked |

Moonshot's own API retired `kimi-k2.5` on 2026-08-31.

## 4. Kimi reply → decision

Kimi is asked for `{"decision": "accept" | "reject" | "unsure", "description": "..."}`.

| Case | `decision` | `description` | Final `box` | Final `point` |
|---|---|---|---|---|
| Valid reply, `accept` | `accept` | Kimi's description | **Exactly** the base `box` | The base `point` |
| Valid reply, `reject` | `reject` | `No differences identified.` | `null` | `null` |
| Valid reply, `unsure` | `unsure` | `No differences identified.` | `null` | `null` |
| Reply isn't readable JSON, or has a value outside the three | `unsure` | `No differences identified.` | `null` | `null` |
| No base box, so Kimi isn't called | `reject` | `No differences identified.` | `null` | `null` |

Kimi never changes the coordinates. It can only keep or drop the detector's box.

## 5. Output files (Test 2)

**`responses.jsonl`** is the runner's own record, one line per pair: `sample_id`, `box`, `called`, `raw_response`, `decision`, `description`, `usage` (tokens and cost), `finish_reason`, `provider`.

**`verification.jsonl`** is the decision file `score-vlm` reads:

| Field | Value |
|---|---|
| `sample_id` | Same as the base prediction |
| `decision` | From section 4 |
| `box` | Base `box` if `accept`, otherwise `null` |
| `description` | From section 4 |
| `source_prediction_sha256` | SHA-256 of the whole `fc-siam-base.jsonl` file that Kimi was run on |

**`fc-siam-kimi.jsonl`** is the official DiffSpot prediction file. It contains exactly these six fields:

| Field | Value |
|---|---|
| `id` | `sample_id` |
| `split` | `easy`, `medium`, `hard` or `no_diff`, joined from `annotations.jsonl` **after** all Kimi calls have finished |
| `model` | `selected-fc-siam+kimi-k2.5` |
| `prompt_version` | `rl-verify-v1` |
| `raw_response` | Kimi's full reply, or `No detector proposal; Kimi was not called.` |
| `prediction` | The same text as `description` |

**Checks (from `score-vlm`)**
- Every ID appears exactly once in each file.
- An accepted box equals the base box.
- A rejected or unsure row has no box and the no-difference text.
- `description` equals `prediction`.
- The hash matches the base file.
- `raw_response` and `prediction` aren't empty.

A failed API call is an unfinished run. It's never written as a no-change answer.

## 6. Frozen settings entry

Entry `selected-fc-siam` in the team's copy of `configs/diffspot.example.json`:

| Field | Value |
|---|---|
| `checkpoint_sha256` | `shasum -a 256 runs/20261111-210302/ckpt_best.pt` |
| `seed_policy` | Single checkpoint, run 3 best pass 33; inference has no randomness |
| `validation_metrics_file` | `docs/results/fc_siam_diff/fc_siam_diff_val.metrics.json` |
| `preprocessing` | RGB, both screenshots padded white at the bottom and right to the larger size, predicted on 192 px squares around differing pixels (`INPUT_MODE = 'diff_squares'`) |
| `box_rule` | Box around all pixels with confidence > 0.5; point at the most confident pixel; none if nothing is flagged |
| `supports_type` | `true` |

## 7. Example

These values are made up to show the mapping. They aren't results.

```json
{"sample_id": "00ded9e6-…", "score": 0.71, "op_pred": "font_weight", "box": [416, 348, 78, 11], "point": [446, 352]}
```

1. **Detection:** `0.71 ≥ 0.0816`, so Test 1 counts it as a regression.
2. **Kimi's input:** the crop region is `[384, 316, 526, 391]` (left, top, right, bottom), sent with both overviews.
3. **Kimi's reply:** `{"decision": "accept", "description": "The subtitle text is bolder."}`
4. **Verification row:** `{"sample_id": "00ded9e6-…", "decision": "accept", "box": [416, 348, 78, 11], "description": "The subtitle text is bolder.", "source_prediction_sha256": "…"}`
5. **Official row:** `{"id": "00ded9e6-…", "split": "<from annotations>", "model": "selected-fc-siam+kimi-k2.5", "prompt_version": "rl-verify-v1", "raw_response": "<full reply>", "prediction": "The subtitle text is bolder."}`

## 8. To agree with the team before the full run

- **No-box rule:** pairs without a box are answered `reject` without calling Kimi.
- **Image labels:** the four labels in section 3 are added before the images. The prompt file itself is unchanged.
- **Host and thinking:** OpenRouter pinned to Bedrock, with thinking off, since Moonshot no longer serves K2.5.

All three must be recorded in the frozen settings before any full-DiffSpot run.

## 9. Built so far

| Part | Status |
|---|---|
| Kimi runner (`scripts/diffspot_kimi.py`), sections 3–4 and `responses.jsonl` / `verification.jsonl` | Built; 44-pair cost check run with stand-in boxes |
| FC-Siam-diff adapter on `inference.jsonl` (section 2) | Not built; needs the checkpoint on the ThinkStation |
| Writing the official file and joining `split` (section 5) | Not built |
