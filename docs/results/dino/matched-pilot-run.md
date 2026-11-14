# Matched DINOv3 pilot runner

`scripts/run_dinov3_pilot.py` repeats the frozen 13 trios plus the separate
position supplement. It reads the existing manifest and DINOv2 artifacts; it
does not sample new pages, train, or evaluate full pages. It checks metadata,
split, manifest and normalization compatibility before inference. The original
notebook's tested crop/distance functions are loaded by AST without executing
notebook cells; keep the notebook alongside the script when transferring it.

Outputs go to ignored `outputs/dino_fit/dinov3/pilot_13/`: 14 native-map archives,
14 comparison figures, raw JSON/CSV statistics, and a DINOv2/DINOv3 comparison
CSV. The 78 baseline summaries and six supplementary summaries carry explicit
subset labels. Figures retain the DINOv2 pooled display scale. A nonempty result
directory is rejected to preserve previous runs; use a separate study root
containing the frozen baseline inputs for an independent rerun.

Local cached-weight run, from the repository's PowerShell terminal:

```powershell
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe scripts/run_dinov3_pilot.py --data-root D:\csc409-data\d2c_v3 --device cpu
Remove-Item Env:HF_HUB_OFFLINE
```

The offline setting is only suitable after the pinned weights are cached.
For online Windows downloads, omit it and pass `--system-trust` with `truststore`
installed. Authentication is stored by Hugging Face, never in the script.

For a fresh lab run, first transfer the latest source archive using
`scripts/prepare_lab_transfer.ps1 -SkipData`. The dataset is already on the lab
machine. Do not copy local `dinov3/pilot_13` results into the intended fresh run
directory: preserve them elsewhere if doing a separate GPU measurement.

```bash
export LAB_ROOT="/virtual/.cache/$USER/csc490"
export HF_HOME="$LAB_ROOT/huggingface"
export TMPDIR="$LAB_ROOT/tmp"
export MPLCONFIGDIR="$LAB_ROOT/matplotlib"
cd "$LAB_ROOT/csc490-project"
"$LAB_ROOT/.venv/bin/python3" -c 'from huggingface_hub import login; login(add_to_git_credential=False)'
"$LAB_ROOT/.venv/bin/python3" scripts/run_dinov3_pilot.py \
  --data-root "$LAB_ROOT/data/d2c_v3" --device cuda
```

Local authentication does not authenticate the remote account. Use the same
approved Hugging Face account on the lab host. The runner uses pinned official
weights, float32 SDPA, frozen parameters, native 448 px crops and no resizing.
These ground-truth-guided crops measure feature sensitivity; they do not yield
held-out detection accuracy or establish a general mutation-type ranking.
