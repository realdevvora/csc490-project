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

The local dataset root is `D:\csc409-data\d2c_v3`. Change `DATA_ROOT` in the
notebook on the lab machine. The official split always comes from the repository,
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

## Following checkpoints

1. Load frozen `facebook/dinov2-base`, inspect its 32 x 32 patch grid, and produce
   cosine/L2 distance maps for one trio, retaining raw statistics.
2. Run the fixed sample and record preliminary operator-specific observations,
   including benign activation and no-change distances.
3. Once weight access is available, repeat with
   `facebook/dinov3-vitb16-pretrain-lvd1689m` using identical crops and its 28 x 28
   grid. Exclude CLS/register tokens from spatial comparisons.

Use consistent colour scales and preserve unscaled distances. Localization points
must map back to original screenshot coordinates and follow the shared scorer's
reflow convention. No head training, full-page tiling, DiffSpot adapter, or shared
metric changes are part of these checkpoints.
