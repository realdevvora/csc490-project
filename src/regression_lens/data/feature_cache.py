"""On-disk cache of frozen encoder hidden states, one file per screenshot.

When the first ``layer`` encoder blocks are frozen, their output for a given
screenshot never changes, so it is computed once and reused every epoch and
every run with the same backbone and tile geometry. Training then only runs
the remaining (trainable or frozen) blocks and the heads.

Each screenshot is cached at its own height. A pair whose other image is
taller is padded with the cached hidden states of an all-white tile, which is
exactly what the uncached path encodes for its white padding rows.

Layout::

  <root>/<checkpoint with / as __>/layer<L>_w<width>/
    meta.json                   backbone, layer and geometry the cache is valid for
    white.pt                    hidden states of one all-white tile, [S, d]
    screenshots/d2c_000.pt      [T, S, d] per screenshot, in the encoder's output dtype

Requires the ``uiclip`` extra (torch).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset

from regression_lens.data.pairs import first_item, load_tiles

if TYPE_CHECKING:
    from regression_lens.models.uiclip import SiameseDetector

CACHE_FORMAT_VERSION = "1"


class FeatureCache:
    """Read and write cached hidden states for one backbone, layer and geometry."""

    def __init__(self, directory: Path, meta: Mapping[str, Any]) -> None:
        self.directory = directory
        self.meta = dict(meta)
        self._white: torch.Tensor | None = None

    @classmethod
    def open(
        cls,
        root: str | Path,
        checkpoint: str,
        layer: int,
        tile: int,
        patch: int,
        cols: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> FeatureCache:
        """Open (creating if needed) the cache for this exact configuration."""

        width = cols * tile
        directory = Path(root) / checkpoint.replace("/", "__") / f"layer{layer}_w{width}"
        meta = {
            "format_version": CACHE_FORMAT_VERSION,
            "checkpoint": checkpoint,
            "layer": layer,
            "tile": tile,
            "patch": patch,
            "cols": cols,
            "mean": list(mean),
            "std": list(std),
        }
        meta_path = directory / "meta.json"
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            if existing != meta:
                raise ValueError(f"{meta_path} was built with different settings:\n{existing}\nvs\n{meta}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, indent=2))
        return cls(directory, meta)

    @property
    def layer(self) -> int:
        return self.meta["layer"]

    def path(self, image_path: str) -> Path:
        return self.directory / Path(image_path).with_suffix(".pt")

    def missing(self, image_paths: Iterable[str]) -> list[str]:
        paths = [p for p in image_paths if not self.path(p).exists()]
        if not (self.directory / "white.pt").exists():
            paths.append("white")
        return paths

    def save(self, image_path: str, hidden: torch.Tensor) -> None:
        """Write atomically so an interrupted build never leaves a corrupt file."""

        path = self.directory / "white.pt" if image_path == "white" else self.path(image_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(hidden.contiguous().cpu(), temporary)
        os.replace(temporary, path)

    def load(self, image_path: str) -> torch.Tensor:
        return torch.load(self.path(image_path), weights_only=True)

    def white(self) -> torch.Tensor:
        if self._white is None:
            self._white = torch.load(self.directory / "white.pt", weights_only=True)
        return self._white

    def load_padded(self, image_path: str, n_tiles: int) -> torch.Tensor:
        """Cached ``[T, S, d]`` hidden states, padded with white tiles to ``n_tiles``."""

        hidden = self.load(image_path)
        if hidden.shape[0] > n_tiles:
            raise ValueError(f"{image_path}: cached {hidden.shape[0]} tiles, pair needs {n_tiles}")
        padding = self.white().to(hidden.dtype).expand(n_tiles - hidden.shape[0], *hidden.shape[1:])
        return torch.cat([hidden, padding])


class _ScreenshotTiles(Dataset):
    def __init__(self, images: list[tuple[str, float, int]], data_root: Path, cache: FeatureCache) -> None:
        self.images, self.data_root = images, data_root
        meta = cache.meta
        self.width, self.tile = meta["cols"] * meta["tile"], meta["tile"]
        self.mean = torch.tensor(meta["mean"]).view(3, 1, 1)
        self.std = torch.tensor(meta["std"]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        path, scale, rows = self.images[index]
        return path, load_tiles(self.data_root / path, scale, rows, self.width, self.tile, self.mean, self.std)


@torch.no_grad()
def build_feature_cache(
    cache: FeatureCache,
    model: SiameseDetector,
    images: Mapping[str, tuple[float, int]],
    data_root: str | Path,
    device: torch.device,
    workers: int = 4,
) -> None:
    """Encode every screenshot in ``images`` that is not cached yet.

    ``images`` maps a screenshot path to ``(scale, tile rows)``, as returned by
    ``PairDataset.images()``. Already-cached screenshots are skipped, so an
    interrupted build can simply be rerun.
    """

    if cache.layer != model.cache_layer:
        raise ValueError(f"cache is for layer {cache.layer}, model caches layer {model.cache_layer}")
    model.eval()
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    todo = cache.missing(images)

    if "white" in todo:
        todo.remove("white")
        meta = cache.meta
        white = torch.ones(1, 3, meta["tile"], meta["tile"])
        white = (white - torch.tensor(meta["mean"]).view(3, 1, 1)) / torch.tensor(meta["std"]).view(3, 1, 1)
        with torch.autocast(**autocast):
            cache.save("white", model.frozen_hidden_states(white.to(device))[0])

    if not todo:
        return
    # A private generator keeps the build from consuming the global RNG, so a
    # run that builds the cache shuffles exactly like one that reuses it.
    loader = DataLoader(
        _ScreenshotTiles([(p, *images[p]) for p in todo], Path(data_root), cache),
        batch_size=1, collate_fn=first_item, num_workers=workers, generator=torch.Generator(),
    )
    for done, (path, tiles) in enumerate(loader, start=1):
        with torch.autocast(**autocast):
            cache.save(path, model.frozen_hidden_states(tiles.to(device)))
        if done % 500 == 0 or done == len(todo):
            print(f"feature cache: {done}/{len(todo)} screenshots encoded")
