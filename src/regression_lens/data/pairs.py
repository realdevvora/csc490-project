"""PyTorch dataset of tiled reference/candidate screenshot pairs.

Screenshots are tall (1280 px wide, up to ~3300 px high), so instead of
squashing a page into one 224x224 input, each page is resized to a width that
is a whole number of tiles, padded with white to a whole number of tile rows,
and cut into aligned square tiles. Reference and candidate use the same scale
factor, so tile ``i`` of both images covers the same region of the page.

With a ``FeatureCache``, items carry cached encoder hidden states instead of
pixels (see ``regression_lens.data.feature_cache``).

Requires the ``uiclip`` extra (torch, Pillow).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from regression_lens.data.feature_cache import FeatureCache

LABELS = ("regression", "no_change", "benign")


def load_records(data_root: str | Path) -> list[dict[str, Any]]:
    """Load every pair record from ``master_dataset_with_benign.json``."""

    return json.loads((Path(data_root) / "master_dataset_with_benign.json").read_text())


def operator_names(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return the sorted mutation operators, the class order of the type head."""

    return sorted({record["operator"] for record in records if record["operator"]})


def first_item(batch: list[Any]) -> Any:
    """DataLoader collate function for batch size 1 with variable tile counts."""

    return batch[0]


def to_tiles(image: torch.Tensor, tile: int) -> torch.Tensor:
    """Cut a ``[3, H, W]`` image into row-major ``[T, 3, tile, tile]`` tiles."""

    tiles = image.unfold(1, tile, tile).unfold(2, tile, tile)  # [3, R, C, t, t]
    return tiles.permute(1, 2, 0, 3, 4).reshape(-1, 3, tile, tile)


def cells_to_patches(cells: torch.Tensor, cells_per_tile: int) -> torch.Tensor:
    """Regroup a page-level ``[R*g, C*g]`` grid into tile-major ``[T, g*g]``.

    This matches the order of a ViT's patch tokens for the tiles from
    ``to_tiles``. ``patches_to_cells`` is the inverse.
    """

    g = cells_per_tile
    grouped = cells.unfold(0, g, g).unfold(1, g, g)  # [R, C, g, g]
    return grouped.reshape(-1, g * g)


def patches_to_cells(patches: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Turn per-tile ``[T, g*g]`` patch values back into a ``[R*g, C*g]`` page map."""

    rows, cols = grid
    g = math.isqrt(patches.shape[-1])
    return patches.view(rows, cols, g, g).permute(0, 2, 1, 3).reshape(rows * g, cols * g)


def image_rows(height: int, scale: float, tile: int) -> int:
    """Tile rows needed for an image of ``height`` original pixels at ``scale``."""

    return max(1, math.ceil(round(height * scale) / tile))


def load_tiles(
    path: str | Path,
    scale: float,
    rows: int,
    width: int,
    tile: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Resize by ``scale``, pad with white to ``rows`` x ``width``, normalize and tile."""

    image = Image.open(path).convert("RGB")
    size = (round(image.width * scale), round(image.height * scale))
    height = rows * tile
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image.resize(size, Image.BICUBIC), (0, 0))
    pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
    pixels = pixels.view(height, width, 3).permute(2, 0, 1).float() / 255
    return to_tiles((pixels - mean) / std, tile)


class PairDataset(Dataset):
    """Tiled screenshot pairs with image, patch and mutation-type targets.

    Each item is a dict with:
      ref, cand      ``[T, 3, tile, tile]`` normalized tiles, or, with a
                     feature cache, ``[T, S, d]`` cached encoder hidden states
      cached         whether ``ref``/``cand`` are cached hidden states
      patch_target   ``[T, g*g]`` 1 where a patch overlaps ``diff_bbox``
                     (regressions only; benign and no-change pairs are all 0)
      grid           ``(rows, cols)`` of tiles
      scale          resize factor from original pixels to tile pixels
      is_regression  1.0 or 0.0
      op             operator class index, or -100 for non-regressions
      record         the original metadata record
    """

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        data_root: str | Path,
        pages: Iterable[str],
        operators: Sequence[str],
        tile: int = 224,
        patch: int = 32,
        cols: int | None = None,
        mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
        std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
        labels: Iterable[str] = LABELS,
        feature_cache: FeatureCache | None = None,
    ) -> None:
        pages, labels = set(pages), set(labels)
        self.records = [
            r for r in records if r["parent_id"] in pages and r["label"] in labels
        ]
        self.data_root = Path(data_root)
        self.operators = list(operators)
        self.tile, self.patch = tile, patch
        self.cols = cols or math.ceil(1280 / tile)
        self.width = self.cols * tile
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.feature_cache = feature_cache

    def __len__(self) -> int:
        return len(self.records)

    def _scale(self, record: Mapping[str, Any]) -> float:
        return self.width / record["size_ref"][0]

    def images(self) -> dict[str, tuple[float, int]]:
        """Every screenshot this dataset reads, mapped to ``(scale, own tile rows)``."""

        images = {}
        for r in self.records:
            scale = self._scale(r)
            images[r["reference_path"]] = (scale, image_rows(r["size_ref"][1], scale, self.tile))
            images[r["candidate_path"]] = (scale, image_rows(r["size_cand"][1], scale, self.tile))
        return images

    def _image(self, relative_path: str, scale: float, rows: int) -> torch.Tensor:
        if self.feature_cache is None:
            return load_tiles(self.data_root / relative_path, scale, rows, self.width,
                              self.tile, self.mean, self.std)
        return self.feature_cache.load_padded(relative_path, rows * self.cols)

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self.records[index]
        scale = self._scale(r)
        rows = max(image_rows(r["size_ref"][1], scale, self.tile),
                   image_rows(r["size_cand"][1], scale, self.tile))

        cells = torch.zeros(rows * self.tile // self.patch, self.width // self.patch)
        if r["label"] == "regression":
            x, y, w, h = (v * scale / self.patch for v in r["diff_bbox"])
            cells[int(y) : math.ceil(y + h), int(x) : math.ceil(x + w)] = 1

        return {
            "ref": self._image(r["reference_path"], scale, rows),
            "cand": self._image(r["candidate_path"], scale, rows),
            "cached": self.feature_cache is not None,
            "patch_target": cells_to_patches(cells, self.tile // self.patch),
            "grid": (rows, self.cols),
            "scale": scale,
            "is_regression": float(r["label"] == "regression"),
            "op": self.operators.index(r["operator"]) if r["operator"] else -100,
            "record": r,
        }
