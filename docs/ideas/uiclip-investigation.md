# UIClip / CLIP / SigLIP investigation

Track 4 of the [model investigation plan](model-investigation-plan.md).

## Model

UIClip (Wu et al., UIST 2024, [arXiv 2404.12500](https://huggingface.co/papers/2404.12500)) is OpenAI CLIP ViT-B/32 fine-tuned on JitterWeb (synthetic web pages with injected design defects) to score UI design quality. Only weights are public, on [Hugging Face (BIG Lab, MIT licence)](https://huggingface.co/biglab); there is no official training repo. Checkpoints load as standard `CLIPModel` weights and use the `openai/clip-vit-base-patch32` preprocessing:

- `biglab/uiclip_jitteredwebsites-2-224-paraphrased`
- `biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs`
- `biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs` (default)

UIClip scores whole screenshots, so on its own it cannot localize a change, and resizing a 1280 px page to 224 px removes 1–2 px changes. We therefore:

1. resize both screenshots to a whole number of tiles across (1344 px = 6 × 224) with the same scale factor;
2. cut them into aligned 224 × 224 tiles;
3. use the encoder's 7 × 7 patch tokens per tile, fused as `[a, b, |a−b|, a·b]`;
4. predict a per-patch change map, an image-level regression score, and the mutation operator.

```text
ref tiles  ──┐                           ┌─> patch logits ─> change map (localization)
             ├─ UIClip vision (shared) ──┤─> image logit (regression vs not)
cand tiles ──┘   patch tokens [T,49,768] └─> type logits (13 operators)
```

## Main findings

**Setup of these results:**
- Every experiment was trained for 10 epochs with seeds 0, 1 and 2, using the feature cache.
- Each run is represented by its best-AUROC checkpoint, and values are mean ± standard deviation across the 3 seeds.
- Evaluation is on the validation split: 73 pages, 1,108 pairs (889 regression, 146 benign, 73 no-change).
- Scoring used `scripts/score_predictions.py`, and the full table is in `outputs/comparison/comparison.md`.

| Experiment | AUROC reg/benign | Benign FPR @95% recall | F1 | Benign FPR | No-change FPR | Type macro-F1 | Pointing | Mean IoU | Recall, reflowed changes |
|---|---|---|---|---|---|---|---|---|---|
| UIClip, frozen (`uiclip_cal`) | 0.993 ± 0.002 | 0.009 | 0.983 | 0.041 | 0.032 | 0.36 ± 0.03 | 0.44 | 0.20 | 0.82 |
| UIClip, frozen, no calibration (`uiclip_nocal`) | 0.994 ± 0.001 | 0.002 | 0.983 | 0.030 | 0.055 | 0.35 ± 0.03 | 0.47 | 0.21 | 0.81 |
| CLIP, frozen (`clip_cal`) | 0.992 ± 0.002 | 0.009 | 0.979 | 0.039 | 0.027 | 0.34 ± 0.06 | 0.40 | 0.21 | 0.78 |
| UIClip, no benign training pairs (`uiclip_cal_nobenign`) | 0.622 ± 0.057 | 0.993 | 0.919 | 1.000 | 0.041 | 0.38 ± 0.01 | 0.46 | 0.18 | 1.00 |
| UIClip, last 4 blocks fine-tuned (`uiclip_cal_ft4`) | **0.999 ± 0.001** | 0.009 | **0.995** | 0.034 | 0.041 | **0.45 ± 0.00** | **0.47** | 0.19 | **1.00** |

1. **Fine-tuning the last 4 encoder blocks gives the best model.**
   - Compared with the frozen encoder:
     - AUROC rises from 0.993 to 0.999 and F1 from 0.983 to 0.995.
     - Type macro-F1 rises from 0.36 to 0.45.
     - Recall on reflowed changes rises from 0.82 to 1.00, and on the layout family from 0.94 to 1.00.
   - Costs:
     - Training takes about 4× as long: 37–41 min per run instead of about 10.
     - Mean IoU is slightly lower (0.19 vs 0.20).

2. **Benign training pairs are essential.**
   - Trained only on regressions and identical pairs, the model flags every Firefox/WebKit pair (benign FPR 1.00, AUROC 0.62), while still catching 99% of regressions.
   - In effect it learns "any difference is a regression".
   - This is the clearest answer so far to the regression-only vs. regression + benign question in the [investigation plan](model-investigation-plan.md).

3. **UI-specific pretraining helps slightly at best.**
   - With frozen encoders, UIClip is ahead of plain CLIP on every metric: F1 0.983 vs 0.979, pointing 0.44 vs 0.40, type 0.36 vs 0.34, and reflow recall 0.82 vs 0.78.
   - Most gaps are within one to two standard deviations of seed noise, so this is weak evidence.
   - CLIP has not been fine-tuned yet, so we don't know whether the gap survives fine-tuning.

4. **Self-calibration fixes the identical-pair problem without hurting detection.**
   - No-change FPR falls from 0.055 to 0.032, while AUROC (0.993 vs 0.994) and F1 (0.983 for both) are unchanged.
   - The remaining no-change false positives are mostly pairs that aren't actually identical (see below).
   - Pointing accuracy is lower with calibration (0.44 vs 0.47). Whether that's a real cost is still open.

5. **Detection is easy on this data; localization and change type are hard.**
   - **Detection:** every model trained with benign pairs reaches AUROC ≥ 0.99. Recall is about the same for small, medium and large mutations (0.96–0.98 frozen, ≥ 0.99 fine-tuned).
   - **Localization:** pointing accuracy stays at 0.40–0.47 and IoU at about 0.2. The change map is a coarse 32 px patch grid, and CLIP-style patch features are known to localize poorly.
   - **Change type:**
     - Colour and border changes are recognised well; F1 for `background_color` is 0.84–0.87 and for `border_width` and `border_radius` 0.66–0.80.
     - Subtle spacing and typography changes are not; F1 for `font_size`, `position` and `margin` is 0.09–0.25.

6. **Seed variance is large relative to the differences between models.**
   - The same configuration's benign FPR ranges from 0.014 to 0.068 across seeds.
   - Compare models by mean ± spread over several seeds, never by single runs.

**Caveats:**
- These are validation numbers: the best epoch and the threshold were both selected on validation, so they are optimistic.
- The split is still the placeholder 85/15, and there is no test split yet.
- Generalization to DiffSpot is untested.
- 25 of the dataset's 484 `no_change` pairs are not actually identical.
- SigLIP has not been run.

**Next steps:**
- Evaluate the frozen best model on DiffSpot.
- Add a page-level test split.
- Fine-tune CLIP's last 4 blocks to settle the UIClip vs. CLIP question.
- Improve localization, e.g. with finer patches (SigLIP patch16 or higher-resolution tiles).
- Pass the change map and boxes to the VLM description stage.

## Code

| Path | Contents |
|---|---|
| `src/regression_lens/data/splits.py` | page-level split; placeholder seeded 85/15 (411/73 pages) |
| `src/regression_lens/data/pairs.py` | `PairDataset`: tiling, patch targets from `diff_bbox` |
| `src/regression_lens/data/feature_cache.py` | on-disk cache of frozen encoder features |
| `src/regression_lens/models/uiclip.py` | `SiameseDetector`, `detector_loss`, checkpoint ids |
| `src/regression_lens/evaluation/metrics.py` | precision/recall/F1, FPR, macro-F1, box IoU, pointing accuracy |
| `src/regression_lens/evaluation/scoring.py` | shared scoring of prediction files |
| `scripts/train_uiclip.py` | training and evaluation CLI, writes `outputs/<exp>/` |
| `scripts/score_predictions.py` | scores and compares any models' prediction files |

## Setup

You need Python 3.10 or newer. A GPU is optional: the training script uses an NVIDIA GPU (CUDA) if PyTorch can see one and falls back to the CPU otherwise. It doesn't use Apple's MPS backend, so Macs run on the CPU.

1. **Create and activate a virtual environment** in the project folder:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Linux, macOS, WSL
   # .\.venv\Scripts\Activate.ps1     # Windows PowerShell
   ```

   On Debian/Ubuntu, `python -m venv` fails until the `python3-venv` package is installed (e.g. `sudo apt install python3.12-venv`).

2. **Install PyTorch for your hardware.** Use the selector at [pytorch.org/get-started](https://pytorch.org/get-started/locally/), which gives the right command for CPU-only, a CUDA version, or ROCm. Install it before the next step, otherwise pip picks a default build that may not match your GPU.

3. **Install the project and its model dependencies:**

   ```bash
   pip install -e ".[dev,uiclip]"
   python -m pytest                   # should pass without a GPU
   ```

4. **Get the data.** Unzip `d2c_v3.zip` so that `d2c_v2/master_dataset_with_benign.json` and `d2c_v2/screenshots/` exist in the project folder, or point `--data-root` at another location.

Practical notes:

- **Disk:** each feature cache takes about 53 GB for UIClip/CLIP (see [Feature caching](#feature-caching)), and the full experiment batch builds three of them. Training without `--cache-features` needs no extra disk.
- **CPU-only:** the pipeline works but is much slower. Use the feature cache, and check the setup first with a smoke run (`--limit-pages 5 --epochs 1`).
- **Shell:** `scripts/run_uiclip_experiments.sh` is a bash script. On Windows, run it from WSL or Git Bash, or run its `train_uiclip.py` commands one by one.
- **First run:** model weights download from Hugging Face on first use (about 600 MB per backbone), so the first run needs internet access.

### Machine used for these experiments

All results in this document were produced on:

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM (driver 610.88) |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| RAM | 30 GiB available to WSL |
| OS | Ubuntu 24.04.5 LTS on WSL2 (Windows), kernel 6.6.87.2 |
| Disk | 1 TB volume, about 800 GB free during the runs |
| Software | Python 3.12.3, PyTorch 2.11.0 (CUDA 12.8 build), transformers 5.22.0 |

RTX 50-series (Blackwell) GPUs need a PyTorch build for CUDA 12.8 or newer. On this machine, step 2 was:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

On CUDA GPUs, training uses bf16 mixed precision, which runs natively on Ampere (RTX 30-series) and newer GPUs; older GPUs may run it slowly. The timings quoted below (about 50–60 s per epoch with the feature cache) are from this machine.

## Runs

```bash
python scripts/train_uiclip.py --exp smoke --limit-pages 5 --epochs 1          # sanity check
python scripts/train_uiclip.py --exp uiclip_frozen                             # heads only
python scripts/train_uiclip.py --exp uiclip_ft4 --train-last-n 4               # + last 4 blocks
python scripts/train_uiclip.py --exp uiclip_frozen_nobenign --exclude-benign   # benign ablation
python scripts/train_uiclip.py --exp clip_frozen --checkpoint openai/clip-vit-base-patch32
python scripts/train_uiclip.py --exp siglip_frozen --checkpoint google/siglip-base-patch16-512
```

### Experiment batch

`scripts/run_uiclip_experiments.sh` runs the planned comparisons (calibrated and uncalibrated UIClip, CLIP, no-benign training, and fine-tuning of the last 4 blocks). It trains each for 10 epochs with seeds 0–2, then writes `outputs/comparison/comparison.md`. See the script header for how to start, monitor and stop it.

### Feature caching

When the first encoder blocks are frozen, their output for a screenshot never changes. `--cache-features` computes that output once per screenshot, saves it under `data/processed/features/`, and trains from it:

```bash
python scripts/train_uiclip.py --exp uiclip_frozen --cache-features     # first run also builds the cache
python scripts/train_uiclip.py --exp uiclip_ft4 --train-last-n 4 --cache-features   # caches the first 8 blocks
```

- **Speed:** on 20 pages × 2 epochs, the uncached run took 64 s. The cached run took 27 s the first time (including building the cache) and 14 s once the cache existed, most of it model loading.
- **Disk:** about 53 GB per UIClip/CLIP configuration, stored as fp32 to keep small reference/candidate differences exact. SigLIP-512 would need about 285 GB, so it's better run without the cache.
- **Separate caches:** each backbone and each `--train-last-n` value gets its own cache. An interrupted build resumes where it stopped.
- **Exactness:** features are bit-identical to recomputing them. The one exception is bf16 rounding in white padding tiles: when a candidate is taller than its reference, the reference is padded, and the padding tile is encoded on its own. This affects about 1.4% of pairs. Training then drifts slightly from an uncached run, by less than changing `--seed` does.

**Seed noise is large.** On 20-page runs, `--seed` 0/1/2 gave AUROC 0.77/0.89/0.77. Compare models across several seeds and the full validation set, not single short runs.

The first time it runs, the script writes the placeholder split to `data/processed/splits_placeholder_85_15.json`, and later runs reuse it. To use the team split, pass `--split path/to/splits_85_15.json`. The file must contain `{"train": [page ids], "val": [page ids]}`.

## Identical screenshots flagged as regressions

### Observation

The first full run (`outputs/uiclip_frozen_s1_5_epochs/`: UIClip frozen, seed 1, 5 epochs, without self-calibration) flagged 3 of the 73 validation `no_change` pairs as regressions (4.1% FPR). Those pairs are meant to be pixel-identical. Across the 73 pairs the model produced 68 different scores, from 0.0009 to 0.995, even though every pair compared a page with itself.

### Cause

The heads see the fused features `[a, b, |a−b|, a·b]`. For identical images `|a−b|` is exactly zero, but `a`, `b` and `a·b` still carry the page's content, and the image head pools them with a max and an attention-weighted average. Nothing forced the output for an identical pair to be a fixed "no change" value, so the score partly reflected what a page looks like rather than what changed. Some pages simply looked like regressions to it.

### Fix: self-calibration

The patch and image logits are calibrated against the reference compared with itself:

```text
logit(ref, cand) = head(ref, cand) − head(ref, ref) + learned offset
```

- For identical screenshots the two head outputs cancel exactly, so every identical pair gets the same score, the learned offset, which training pushes low.
- The heads still see page content, so they can learn, for example, that a shifted button matters more than anti-aliasing on text.
- No loss of capacity: any scoring rule that scores all identical pairs the same can be written in this form.
- Cost: the second pass reuses the encoder output and only reruns the small heads.
- Code: `SiameseDetector(self_calibrate=True)`, the default. `--no-self-calibration` trains the old model for comparison. `--eval-only` restores the setting from a run's `config.json`.
- Tests (`tests/test_feature_cache.py`): with calibration, identical pairs from different pages score exactly the offset; without it, they score differently.

### Result

`outputs/uiclip_cal_s1_5_epochs/` uses the same setup as the first run, with calibration on. Both are validated on the same 73 pages:

| Run | AUROC reg/benign | Benign FPR | No-change FPR | Type macro-F1 | Pointing |
|---|---|---|---|---|---|
| Old model (`uiclip_frozen_s1_5_epochs`) | 0.995 | 0.034 | 0.041 (3/73) | 0.28 | 0.47 |
| Calibrated (`uiclip_cal_s1_5_epochs`) | 0.991 | 0.041 | 0.027 (2/73) | 0.33 | 0.44 |

71 of the 73 `no_change` pairs now get exactly the same score. The remaining differences between the two runs are within seed-to-seed noise, so the 10-epoch, 3-seed runs (`cal` vs `nocal`) are needed to tell whether calibration helps or hurts the other metrics.

### Dataset finding: some "no_change" pairs are not identical

The 2 flagged pairs left after calibration differ from their references:

- `d2c_095_identical` is 998 px tall vs 992 px for its reference.
- `d2c_097_identical` has 1,681 changed pixels.

Across the whole dataset, **25 of the 484 `no_change` pairs** have a size mismatch or nonzero `changed_px`, so the same-browser re-render is not always deterministic. These pairs are really same-browser benign variation. They should be relabelled, or regenerated in `CSC490_data_mutation.ipynb`, before `no_change` FPR is treated as a clean metric.

## Evaluation notes

- Each run keeps two checkpoints, `best_auroc/` (best validation AUROC, regression vs benign) and `best_f1/` (best validation F1), each with `model.pt` and `val_predictions.jsonl`. Both prediction files follow the [shared prediction format](../prediction-format.md); see that page for all metrics.
- The decision threshold maximizes F1 on validation and is reported in `metrics.json`. DiffSpot must never be used to pick it.
- FPR is reported separately for `benign` (Firefox/WebKit) and `no_change`. `no_change` pairs are meant to be pixel-identical (25 are not; see above), so `benign` is the meaningful nuisance test.
- IoU compares the largest connected change-map region against `diff_bbox`. Pointing accuracy checks whether the change-map peak falls inside `diff_bbox`, or inside `bbox_after` for reflowed mutations, whose `diff_bbox` covers everything that moved.
- Validation selects both the epoch and the threshold, so validation results are optimistic until an internal test split exists.
- DiffSpot evaluation is not implemented yet.
