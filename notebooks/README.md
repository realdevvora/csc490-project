# Shared DiffSpot walkthrough

[`diffspot_smoke.ipynb`](diffspot_smoke.ipynb) guides **every model owner** through
a common 44-pair sample, inference/import, unchanged validation threshold and
visual box review. It has no DINO-specific model assumption. See
[the model-agnostic guide](../docs/diffspot-smoke.md) for the adapter contract.
Existing DINO exploration documentation follows below.

# DINO dataset-fit exploration

Open `dino_exploration.ipynb` in VS Code with its Python and Jupyter extensions.
The notebook is built in small checkpoints; run the current cells in order and
inspect their outputs before adding the next batch.

## Local environment

From the repository root, in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dino,evaluation,dev]"
```

Select `.venv\Scripts\python.exe` as the notebook kernel. No activation or
global Jupyter-kernel registration is required. Browser-based JupyterLab is
optional; VS Code only needs `ipykernel` in this environment.

The local dataset root is `D:\csc409-data\d2c_v3`; the Linux default is
`~/CSC490/data/d2c_v3`. `REGRESSION_LENS_DATA_ROOT` can override either default
when set for the kernel process. See the [GPU lab setup](../docs/lab-setup.md)
for transfer and environment commands. The official split always comes from the repository,
not a newly generated split. On a CUDA lab machine, install the matching CUDA
builds of PyTorch/torchvision instead of the CPU wheels above.

## Checkpoint 1: fixed sample and model inputs

The team has completed general dataset validation and manual inspection. This
checkpoint prepares inputs specifically for a DINO feature-sensitivity study:

- one regression per operator, selected with seed 490 from official training pages;
- one benign and one no-change control from each selected regression's page;
- the same 448 x 448 crop coordinates for every image in a matched trio;
- original image scale, top-left coordinates, and white padding where needed;
- a saved sample/crop manifest for exact repetition with DINOv3.

The sample includes 13 operators and 39 pair/crop evaluations. Distinct pages are
preferred. Firefox and WebKit controls alternate where available. A no-change or
benign pair can be reused at different locations; the manifest records that crop
context. This is an exploratory sample, not a balanced estimate of performance
for each operator.

Crop centres come from the union of the edited element's before/after boxes.
Large edits can extend beyond the crop. Boxes are displayed clipped to the crop,
but stored in original coordinates. These guided crops measure local feature
response; they do not establish full-page detection performance.

The notebook writes to ignored `outputs/dino_fit/`: a sample manifest, environment
record, and a preview figure. Existing sample/crop choices are never silently
overwritten. To deliberately start a different study, use another output directory.
Keep the same manifest when moving to the lab or comparing DINOv3.

## Checkpoint 2: frozen DINOv2 features and heatmaps

Four additional code cells load pinned `facebook/dinov2-base` weights and run one
matched trio (initially `font_size`). Run the notebook in order. The processor's
resize and centre crop are disabled; checkpoint normalization is retained.

The first run downloads approximately 350 MB to ignored
`outputs/dino_fit/model_cache/`. The model uses float32, evaluation mode, no
gradients, and one image per forward pass. CPU execution uses four threads on this
laptop. There is no optimizer or head.

Expected shapes are `[1, 3, 448, 448]` inputs, `[1, 1025, 768]` output tokens,
and `[1, 1024, 768]` spatial features after removing CLS. Cosine and raw L2
distances between corresponding final-layer patch features become 32 x 32 maps.

The trio's artifacts are saved under `outputs/dino_fit/dinov2/font_size/`:

- `cosine_heatmaps.png` and `l2_heatmaps.png`: reference, candidate, and heatmap
  rows for regression, benign, and no-change pairs;
- `native_maps.npz`: unscaled native maps and valid-patch masks; and
- `statistics.json`: raw summaries, ROI/outside means, original-coordinate peaks,
  shared pointing results for the regression, timings, revisions, and provenance.

Within each figure, all three labels use a shared zero-to-pooled-p99 colour
scale. Saturated values remain intact in the raw maps. Fractional overlap weights
account for boxes smaller than a patch; partially padded patches are excluded.
The matched regression ROI is used for all three pairs, but it is not a defect
annotation for benign/no-change controls. A flat map has no meaningful peak.

Initial observations and their limits are recorded in
[the DINO pilot notes](../docs/results/dino/README.md). Stop after this trio to
understand the tensors and figures before expanding the study.

## Checkpoint 3: fixed 13-operator sample

Four additional code cells evaluate the saved sample manifest using the same
backbone, preprocessing, crop geometry, and distance summaries. Expect 52 image
forwards (13 references plus 39 candidates), with progress after each operator.
The sample remains on official training pages; this does not create a new split.
If your kernel already ran an older checkpoint 2, rerun its distance-summary cell
(code comment `# 7`) first: the common summaries now live in a small helper that
both checkpoints use. Existing displayed notebook outputs are preserved.

Artifacts under `outputs/dino_fit/dinov2/pilot_13/` include:

- `statistics.json` and `statistics.csv`: all 78 distance summaries (39 pairs,
  two distances), timings, geometry, and run provenance;
- one `<operator>_native_maps.npz` per trio;
- `operator_review.csv`: local response, benign response, crop coverage, and
  localization diagnostics side by side;
- `operator_overview.png`: all regression crops and native cosine maps for the
  regression/benign/no-change pairs, using one pilot-wide scale; and
- `display_limits.json`: the frozen DINOv2 plotting limits for later comparison.

The final cell's `REVIEW_OPERATOR` selects a detailed trio figure. Changing it
does not run the model again. `show_pilot_trio('border_radius', scale='trio')`
allows an explicitly labelled local contrast scale while keeping the same scale
for all controls in that figure; the default `scale='pilot'` is shared across all
operators. Raw statistics are unchanged by the display scale.

`ROI visible` measures how much of the known target box is covered by valid crop
patches. The peak-patch-overlap diagnostic explains coarse-grid misses but does
not replace the shared pointing metric. Large targets can fill the crop and make
a pointing hit trivial. Review the figures and geometry rather than treating any
single number as an operator-wide difficulty ranking.

The pilot found zero ROI coverage for the original position crop. A final
follow-up cell keeps that baseline intact and evaluates the same trio using a
separately saved `position_supplement_manifest.json`, centred on its changed
region. Supplementary maps, statistics, and a figure have separate filenames.
The study therefore has 39 baseline cases plus three supplementary crop cases;
it still uses 39 unique pair IDs. This tests input coverage rather than modifying
the team's validated data. Repeat the supplementary manifest with DINOv3 too.

## Checkpoint 4: first-trio DINOv3 comparison (access pending)

The next four code cells are prepared for the same font-size trio, using official
`facebook/dinov3-vitb16-pretrain-lvd1689m` revision
`5931719e67bbdb9737e363e781fb0c67687896bc`. **No pretrained DINOv3 inference has
run yet.** On 2026-11-14, the model's configuration request returned HTTP 401
(`GatedRepoError`), the Hub reported manual access approval, and this computer
had no Hugging Face login configured.

Request access using the account you will use locally at the
[official model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
After approval, authenticate from the repository root:

```powershell
.\.venv\Scripts\hf.exe auth login
```

Enter your token in that terminal prompt, not in the notebook or chat. The
notebook uses the locally stored login; it does not save or print credentials.
Login alone does not grant model access. The load cell stops with instructions
if login or approval is missing, and does not substitute other or random weights.

Rerun the updated distance-summary cell (`# 7`) if the kernel predates checkpoint
4. Its DINOv2 defaults remain unchanged. The functions now accept an explicit
patch size/grid, and `spatial_tokens` validates and removes CLS/register tokens.

Once access is available, run the four DINOv3 cells in order. They verify the
448 x 448 normalized tensors against DINOv2, read the register count from the
model config, retain 784 spatial tokens, and use the same raw distance/ROI math.
The installed Transformers version supplies a fast DINOv3 processor; resizing
and centre cropping remain disabled, and tensor equality with the DINOv2 path is
checked within 1e-6 float32 tolerance before inference.
The final comparison reads saved DINOv2 native maps and plotting limits rather
than rerunning its pilot. The figure shows reference, candidate/ground-truth ROI,
DINOv2 heatmap, and DINOv3 heatmap for each label with one shared scale.

The first-trio comparison has now completed, with actual outputs under
`outputs/dino_fit/dinov3/font_size/`. To repeat all frozen cases without rerunning
the notebook, use [the matched-pilot runner](../docs/results/dino/matched-pilot-run.md).
It preserves the original manifest and separate position supplement. Synthetic
tensor tests validate the math and token indexing; they are not model results.

Use consistent colour scales and preserve unscaled distances. Localization points
must map back to original screenshot coordinates and follow the shared scorer's
reflow convention. No head training, full-page tiling, DiffSpot adapter, or shared
metric changes are part of these checkpoints.
