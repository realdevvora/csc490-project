# VLM diagnosis format

After a detector has been scored, a vision-language model (VLM) gives a second opinion on each of its predictions and writes a short report a QA reviewer can act on. This document defines that report, one JSON object per screenshot pair, and the rules for producing it.

Format version: **1**

## Scope

- **Post-run diagnosis only.** It reads a finished prediction file (see [prediction format](prediction-format.md)) and never changes it. VLM output is not merged into predictions and is not scored by `scripts/score_predictions.py`.
- **Any detector.** The input is the shared prediction format, so it works for every model the team evaluates.
- **Validation pages only** until a model is selected. Test pages and DiffSpot stay untouched (ADR 0001).

## Inputs

For each pair:
- the reference and candidate screenshots;
- the detector's prediction line: `score`, `op_pred`, `box`, `point`;
- the threshold that turns `score` into a regression claim. Use the scorer's `threshold` from the run's `.metrics.json`, and record it.

Ground truth from `master_dataset_with_benign.json` is used **only afterwards**, to judge the VLM. It is never shown to the VLM.

## Rules

1. **Two passes.**
   - **Blind pass:** the VLM sees only the two screenshots and says what changed, if anything.
   - **Check pass:** the VLM is then shown the detector's claim and confirms or denies each part.
   - A confirmation that contradicts the blind answer is a sign of agreeing with whatever it's told.
2. **Check each part of the claim separately:** regression or not, the type, and the box.
3. **"Unsure" is always allowed.** Family-level answers are allowed when the exact type can't be told from pixels (e.g. margin vs padding).
4. **No label leaks.**
   - `sample_id`s contain the true type or browser (e.g. `d2c_000_font_size`, `d2c_000_firefox`).
   - Never put IDs, file names or paths in the prompt, and rename image files before sending them.
5. **Control claims.** On a random 10% of pairs, show a deliberately wrong claim: a wrong type, or a box moved off the change. Record it in `control`. This measures how often the VLM agrees with false claims.
6. **Order swaps.** On a random 10% of pairs, show the candidate before the reference, and record it in `order_swapped`.
7. **What the VLM sees:**
   - crops of both screenshots around the box, padded by `crop.padding_px`, top-aligned (the candidate may be taller after reflow);
   - the same crops with the box drawn on, and without it;
   - a difference image with its contrast boosted, so faint colour and opacity changes are visible;
   - a small whole-page thumbnail, so it can say where on the page the change is.
   - If `box` is null, crop around the region where pixels differ instead.
8. **Dimensions are measured, never estimated** (see `dimensions` below). The VLM never writes a pixel value in free text.
9. **Repeatable:**
   - temperature 0;
   - record the VLM, prompt version and crop settings in every line;
   - keep the raw response.

## File

- [JSON Lines](https://jsonlines.org/): one object per line, UTF-8, one line per pair in the evaluated subset.
- Name: `<detector>_<subset>_vlm.jsonl`, e.g. `fc_siam_diff_val_vlm.jsonl`.

## Fields

Each field's **Source** says who fills it in:
- **harness:** the script running the diagnosis;
- **detector:** copied from the prediction file;
- **VLM:** the model's answer.

### Run information
| Field | Type | Source | Meaning |
|---|---|---|---|
| `sample_id` | string | harness | The pair's `sample_id`. Never shown to the VLM. |
| `format_version` | string | harness | `"1"`. |
| `detector` | string | harness | The detector run, e.g. `"fc_siam_diff/20261111-210302"`. |
| `vlm` | object | harness | `{"name": "...", "version": "..."}`. |
| `prompt_version` | string | harness | Version of the prompt text. |
| `crop` | object | harness | `{"padding_px": int, "thumbnail_width_px": int}`. |
| `order_swapped` | bool | harness | Candidate shown first. |
| `control` | object or null | harness | `{"field": "type" or "box", "shown": <the wrong value shown>}` when the claim was altered on purpose; otherwise null. |
| `raw_response` | string | harness | Path to the saved raw responses of both passes. |

### The claim being checked
| Field | Type | Source | Meaning |
|---|---|---|---|
| `claim.regression` | bool | harness | `score >= threshold`. |
| `claim.score` | number | detector | |
| `claim.threshold` | number | harness | |
| `claim.type` | string or null | detector | `op_pred`. |
| `claim.box` | `[x, y, w, h]` or null | detector | Original screenshot pixels. |
| `claim.point` | `[x, y]` or null | detector | |

### Blind pass
| Field | Type | Source | Meaning |
|---|---|---|---|
| `blind.changed` | `"yes"`, `"no"` or `"unsure"` | VLM | Whether anything visibly changed. |
| `blind.description` | string | VLM | What changed, in one sentence. |

### Checking the claim
Each check holds `agrees` (`"yes"`, `"no"` or `"unsure"`) and `confidence` (0–1).

| Field | Type | Source | Meaning |
|---|---|---|---|
| `verdict_check.agrees` | string | VLM | Agrees with `claim.regression`. |
| `verdict_check.kind` | `"regression"`, `"benign"`, `"no_change"` or `"unsure"` | VLM | Its own call. |
| `verdict_check.confidence` | number | VLM | |
| `type_check.agrees` | string | VLM | Agrees with `claim.type`. |
| `type_check.type` | one of the 13 types, or null | VLM | Its own type, or null if it can only name the family. |
| `type_check.family` | `layout`, `colour`, `typography`, `shape`, `appearance`, or null | VLM | |
| `type_check.confidence` | number | VLM | |
| `box_check.agrees` | string | VLM | The box marks the change. |
| `box_check.on` | `"edited_element"`, `"moved_content"`, `"no_change"` or `"unsure"` | VLM | What the box actually covers. |
| `box_check.confidence` | number | VLM | |

### What a QA reviewer reads
| Field | Type | Source | Meaning |
|---|---|---|---|
| `element` | string | VLM | The affected element in plain words, e.g. "Download PDF button". |
| `before_after` | string | VLM | The change in plain words, e.g. "text changed from dark grey to light grey". No pixel values. |
| `root_cause` | string | VLM | The element whose change caused the difference. |
| `knock_on` | list of strings | VLM | Elements that only moved or resized because of the root cause, e.g. "paragraphs below shifted down". |
| `severity` | `"none"`, `"cosmetic"`, `"readability"` or `"functional"` | VLM | `functional`: overlap, clipped or hidden content, or a smaller click area. `readability`: contrast or legibility. `cosmetic`: visible but harmless. `none`: benign or no change. |
| `user_impact` | string | VLM | One sentence on what a user would notice. |
| `above_fold` | bool or null | harness | The box starts within the first 800 px of the page (viewport height); null without a box. |
| `benign_reason` | string or null | VLM | When it calls the pair benign, why, e.g. "font smoothing differs; layout identical". |
| `evidence` | list of strings | VLM | Visible observations behind its answers, e.g. "the button's right edge is further right". |
| `summary` | string | VLM | One or two sentences for the reviewer. |
| `confidence` | number | VLM | Overall confidence, 0–1. |

### `dimensions` (measured)
All values are in pixels of the original screenshots and computed by the harness, never typed by the VLM.

| Field | Type | Meaning |
|---|---|---|
| `dimensions.changed_region` | `[x, y, w, h]` or null | Box around every pixel that differs between the two screenshots. |
| `dimensions.changed_px` | int | Number of differing pixels. |
| `dimensions.page_before` / `page_after` | `[w, h]` | Screenshot sizes. |
| `dimensions.element_before` / `element_after` | `[x, y, w, h]` or null | The affected element's box in each screenshot, from the VLM's grounding output mapped back to original pixels. Null if the VLM can't locate it. |
| `dimensions.delta` | object or null | `element_after − element_before`: `{"dx", "dy", "dw", "dh"}`. Null if either box is null. |

`element_before` and `element_after` come from the VLM pointing at the element, so they're only as precise as its grounding. Every other dimension is exact.

## Example

```json
{"sample_id": "d2c_018_border_width", "format_version": "1", "detector": "fc_siam_diff/20261111-210302",
 "vlm": {"name": "Qwen3-VL-8B-Instruct", "version": "..."}, "prompt_version": "v1",
 "crop": {"padding_px": 64, "thumbnail_width_px": 320}, "order_swapped": false, "control": null,
 "raw_response": "outputs/vlm/fc_siam_diff/raw/0042.json",
 "claim": {"regression": true, "score": 0.73, "threshold": 0.082, "type": "width", "box": [210, 116, 510, 62], "point": [214, 140]},
 "blind": {"changed": "yes", "description": "The border around the search box is thicker."},
 "verdict_check": {"agrees": "yes", "kind": "regression", "confidence": 0.9},
 "type_check": {"agrees": "no", "type": "border_width", "family": "shape", "confidence": 0.8},
 "box_check": {"agrees": "yes", "on": "edited_element", "confidence": 0.7},
 "element": "Search box", "before_after": "Border went from thin to thick.",
 "root_cause": "Search box border", "knock_on": ["Button to the right shifted slightly"],
 "severity": "cosmetic", "user_impact": "The search box looks heavier but works the same.", "above_fold": true,
 "benign_reason": null,
 "evidence": ["Border lines are visibly thicker on all four sides", "Text inside is unchanged"],
 "summary": "Real regression: the search box border is thicker. The detector found it but named it width instead of border_width.",
 "confidence": 0.8,
 "dimensions": {"changed_region": [212, 118, 506, 58], "changed_px": 3120, "page_before": [1280, 2210], "page_after": [1280, 2210],
                "element_before": [214, 120, 500, 52], "element_after": [212, 118, 504, 56], "delta": {"dx": -2, "dy": -2, "dw": 4, "dh": 4}}}
```

## Judging the VLM

Ground truth is known for every validation pair, so report how far the VLM can be trusted, separately from the detector's scores:

| Measure | Question |
|---|---|
| Error catch rate | On pairs the detector got wrong, how often does the VLM deny the claim? |
| False overturn rate | On pairs the detector got right, how often does the VLM deny it? |
| Control agreement | How often does it confirm a deliberately wrong claim? Lower is better. |
| Type correction | When it denies `claim.type`, is `type_check.type` right more often than the detector? |
| Blind-pass hallucination | On no-change pairs, how often does `blind.changed` say yes? |
| Order sensitivity | Do answers change when the screenshots are swapped? |
| Summary faithfulness | Hand-check 30–50 summaries against the screenshots. |

Report these overall and for the groups where a second opinion matters most:
- benign pairs the detector flagged;
- regressions the detector missed;
- the detector's most common type confusions.
