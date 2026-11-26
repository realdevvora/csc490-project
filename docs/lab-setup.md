# GPU lab setup

The remote machine inspected on 2026-11-14 runs Ubuntu 22.04, Python 3.10.12,
and an RTX 4080 with 16 GB VRAM. Its NVIDIA driver is 580.178.04. Use the same
PyTorch 2.8.0/torchvision 0.23.0 versions as the local notebook, with CUDA wheels
instead of the Windows CPU environment. Driver-reported CUDA 13.0 does not require
installing a CUDA 13 toolkit for these wheels.

## Environment (remote terminal)

**Current host recovery:** the home quota subsequently prevented even the 2 MB
pip bootstrap. Do not retry the home-based environment below on this account
until storage is resolved. Use the machine-local recovery checkpoint in the
quota section first; its new workspace path must also be used for later transfer
commands once verified.

```bash
cd ~/CSC490
python3 -m venv .venv --without-pip
python3 -m pip --python "$HOME/CSC490/.venv/bin/python3" install --no-cache-dir --upgrade pip
source .venv/bin/activate
"$HOME/CSC490/.venv/bin/python3" -m pip --version
"$HOME/CSC490/.venv/bin/python3" -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$HOME/CSC490/.venv/bin/python3" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If `venv`/`ensurepip` is unavailable, capture that error before installing anything
system-wide on the shared lab machine. The final check should report CUDA available
and the RTX 4080. The official PyTorch wheel commands are documented at
https://pytorch.org/get-started/previous-versions/#v280.

### Incomplete environment / quota diagnosis

The first lab attempt hit `Disk quota exceeded`. Subsequent checks showed
`VIRTUAL_ENV=None`, `/usr/bin/python3`, and no pip inside the intended environment.
Do not assume that a bare `python3 -m pip` targets the project environment. The
explicit interpreter paths above avoid relying on activation in a previous shell.
The working system pip can bootstrap an environment without `ensurepip` through
its documented `--python` option:
https://pip.pypa.io/en/stable/topics/python-option/.

Check interpreter prefixes and personal usage before another large installation:

```bash
"$HOME/CSC490/.venv/bin/python3" -c "import sys; print(sys.executable); print(sys.prefix)"
du -sh "$HOME/.local" "$HOME/CSC490/.venv" 2>/dev/null
```

`/student` and `/virtual` are separate filesystems on this lab host. Pip reported
its download cache under `/virtual/.cache/ansar136/pip`, so clearing that cache
would not free a quota on `/student`. `df` reports shared filesystem capacity, not
the account's remaining quota. Fixing interpreter selection does not by itself
guarantee enough quota for CUDA packages. Confirm an approved storage location
before moving large environments, datasets, or feature caches.

### Machine-local recovery checkpoint

The error explicitly names `/student/ansar136/CSC490/.venv/.../pip`. That confirms
the home filesystem quota is exhausted, independently of the interpreter issue.
The existing account cache is on the separate `/virtual` filesystem. Test a
private workspace there without deleting or modifying the failed home environment:

```bash
LAB_ROOT="/virtual/.cache/$USER/csc490"
umask 077
mkdir -p "$LAB_ROOT/tmp"
export TMPDIR="$LAB_ROOT/tmp"
python3 -m venv "$LAB_ROOT/.venv" --without-pip
python3 -m pip --python "$LAB_ROOT/.venv/bin/python3" install --no-cache-dir --upgrade pip
"$LAB_ROOT/.venv/bin/python3" -m pip --version
```

Stop if any step fails; the new pip path must be inside this workspace. After
that bootstrap succeeds, use the explicit interpreter for the GPU installation:

```bash
"$LAB_ROOT/.venv/bin/python3" -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$LAB_ROOT/.venv/bin/python3" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

This is machine-local cache storage, not guaranteed persistent project storage.
Keep the original transfer archives on the PC and back up results outside this
cache. Its successful bootstrap and GPU install are still awaiting user output.
Do not upload the dataset to the full home filesystem.

## Transfer from Windows

`scripts/prepare_lab_transfer.ps1` prepares three files under ignored
`outputs/lab-transfer/`: a Git bundle, a source overlay containing current
uncommitted experiment code and existing DINOv2 results, and the local dataset.
It excludes the Windows environment, model cache, SSH files, and unrelated logs.

From the local repository's PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_lab_transfer.ps1
scp outputs/lab-transfer/repo.bundle outputs/lab-transfer/working-tree.tar.gz outputs/lab-transfer/dataset.tar ansar136@dh2020pc21.utm.utoronto.ca:CSC490/
```

Enter the lab password interactively if prompted. No password is stored in the
repository. In the remote terminal, use a fresh `csc490-project` destination:

```bash
cd ~/CSC490
git clone -b abu-dino-model repo.bundle csc490-project
tar -xzf working-tree.tar.gz -C csc490-project
mkdir -p data
tar -xf dataset.tar -C data
source .venv/bin/activate
cd csc490-project
"$HOME/CSC490/.venv/bin/python3" -m pip install -e ".[dino,evaluation,dev]"
```

For later source-only updates, use `prepare_lab_transfer.ps1 -SkipData` and
transfer only the source overlay rather than rebuilding/sending the dataset.
Review remote changes before overlaying another source snapshot.

The clone supplies history and the overlay supplies uncommitted changes. Check
`git status` before starting jobs. This snapshot does not configure GitHub access;
fetch/push setup can be done separately. The home filesystem's free space is
shared, so inspect personal quota before allocating large feature caches.

## VS Code and notebook

Open `~/CSC490/csc490-project` in the remote VS Code window. Install Python and
Jupyter into the SSH host when prompted. Select `~/CSC490/.venv/bin/python3` as
the notebook kernel. The notebook's remote default dataset location is
`~/CSC490/data/d2c_v3`; `REGRESSION_LENS_DATA_ROOT` can override it when set for
the kernel process. No Windows virtual environment or `D:` paths are required
on the lab.

Use the GPU readout and a short DINOv2 run to benchmark throughput before claiming
that the full experiment matrix will finish by a particular deadline. Long
extraction/training jobs should run through scripts in `tmux` or another lab-
approved persistent session rather than depend on a notebook/SSH window staying
open. DINOv3 remains gated independently of GPU availability.
