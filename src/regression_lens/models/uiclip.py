"""Siamese UIClip/CLIP/SigLIP regression detector.

A shared vision encoder embeds aligned reference and candidate tiles. Their
patch tokens are fused as ``[a, b, |a - b|, a * b]`` and fed to three heads:

  patch head  per-patch change logits -> page change map (localization)
  image head  one regression logit per pair (detection)
  type head   mutation-operator logits (change category)

With ``self_calibrate`` (the default), the patch and image logits are
calibrated against the reference compared with itself:

  logit(ref, cand) = head(ref, cand) - head(ref, ref) + learned offset

so a pair of identical screenshots always gets exactly the offset, the same
"no change" score on every page, while the heads still see page content. Any
scoring rule that treats identical pairs alike can be written this way, so this
removes a failure mode without limiting what the model can learn. The extra
pass reuses the encoder output and only reruns the small heads.

UIClip checkpoints (BIG Lab, CMU; MIT licence) are fine-tuned CLIP ViT-B/32
models at 224 px and load as standard ``CLIPModel`` weights:

  biglab/uiclip_jitteredwebsites-2-224-paraphrased
  biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs
  biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs

``openai/clip-vit-base-patch32`` gives the non-UI CLIP comparison and
``google/siglip-base-patch16-512`` the SigLIP fallback.

Requires the ``uiclip`` extra (torch, transformers).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel

UICLIP_CHECKPOINT = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class EncoderInfo:
    """Input geometry and normalization a backbone expects."""

    tile: int
    patch: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


def encoder_info(checkpoint: str) -> EncoderInfo:
    """Read tile size, patch size and pixel normalization for ``checkpoint``."""

    config = AutoConfig.from_pretrained(checkpoint).vision_config
    try:
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        mean, std = tuple(processor.image_mean), tuple(processor.image_std)
    except (OSError, ValueError):
        # UIClip checkpoints ship without a processor; they use OpenAI CLIP's.
        mean, std = CLIP_MEAN, CLIP_STD
    return EncoderInfo(config.image_size, config.patch_size, mean, std)


class SiameseDetector(nn.Module):
    """Shared-encoder pair model; ``train_last_n`` encoder blocks are trainable."""

    def __init__(
        self,
        checkpoint: str = UICLIP_CHECKPOINT,
        n_operators: int = 13,
        train_last_n: int = 0,
        hidden: int = 256,
        self_calibrate: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint).vision_model
        # CLIP prepends a CLS token to the patch tokens; SigLIP does not.
        self.has_cls = self.encoder.config.model_type.startswith("clip")
        self.train_last_n = train_last_n

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if train_last_n:
            for block in self.encoder.encoder.layers[-train_last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        d = self.encoder.config.hidden_size
        self.fuse = nn.Sequential(
            nn.Linear(4 * d, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
        )
        self.patch_head = nn.Linear(hidden, 1)
        self.image_head = nn.Linear(2 * hidden, 1)
        self.type_head = nn.Linear(hidden, n_operators)
        self.self_calibrate = self_calibrate
        # Score of an identical pair, for each patch and for the whole image.
        # The heads' own biases cancel in the calibration, so these replace them.
        self.patch_offset = nn.Parameter(torch.zeros(()))
        self.image_offset = nn.Parameter(torch.zeros(()))

    def head_parameters(self) -> list[nn.Parameter]:
        heads = [
            p
            for module in (self.fuse, self.patch_head, self.image_head, self.type_head)
            for p in module.parameters()
        ]
        return heads + ([self.patch_offset, self.image_offset] if self.self_calibrate else [])

    def encoder_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.encoder.parameters() if p.requires_grad]

    @property
    def cache_layer(self) -> int:
        """Number of leading encoder blocks that are frozen, and so cacheable."""

        return len(self.encoder.encoder.layers) - self.train_last_n

    # Encoding is split at ``cache_layer``: the frozen part (embeddings and
    # frozen blocks) can be precomputed by ``feature_cache``, and the rest
    # (trainable blocks and final norm) always runs. Both paths share this
    # code, so cached and uncached training see the same features.

    @torch.no_grad()
    def frozen_hidden_states(self, tiles: torch.Tensor) -> torch.Tensor:
        """Hidden states ``[T, S, d]`` after the frozen blocks, for ``[T, 3, tile, tile]`` tiles."""

        hidden = self.encoder.embeddings(tiles)
        if self.has_cls:
            hidden = self.encoder.pre_layrnorm(hidden)
        for block in self.encoder.encoder.layers[: self.cache_layer]:
            hidden = block(hidden, None)
        return hidden

    def finish_encoding(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the trainable blocks on frozen hidden states; return patch tokens ``[T, P, d]``."""

        with torch.set_grad_enabled(self.training and self.train_last_n > 0):
            for block in self.encoder.encoder.layers[self.cache_layer :]:
                hidden = block(hidden, None)
        if self.has_cls:
            return hidden[:, 1:]
        # SigLIP normalizes every token; CLIP's post_layernorm only touches its CLS token.
        return self.encoder.post_layernorm(hidden)

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        """Return patch tokens ``[T, P, d]`` for ``[T, 3, tile, tile]`` tiles."""

        return self.finish_encoding(self.frozen_hidden_states(tiles))

    def forward(
        self, ref: torch.Tensor, cand: torch.Tensor, cached: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score one page.

        ``ref`` and ``cand`` are tiles, or, with ``cached=True``, the output
        of ``frozen_hidden_states`` for them. Returns ``patch_logits [T, P]``,
        ``image_logit []`` and ``type_logits [n_operators]``.
        """

        encode = self.finish_encoding if cached else self.encode
        a, b = encode(ref), encode(cand)
        fused = self._fuse(a, b)
        patch_logits = self.patch_head(fused).squeeze(-1)
        if self.self_calibrate:
            fused_self = self._fuse(a, a)
            patch_logits = patch_logits - self.patch_head(fused_self).squeeze(-1) + self.patch_offset

        # Attention-pool patches by how changed they look, so the type head
        # reads the changed region rather than the whole page.
        weights = patch_logits.flatten().softmax(dim=0)
        pooled = self._pool(fused, weights)
        image_logit = self._image_logit(fused, pooled)
        if self.self_calibrate:
            image_logit = (image_logit - self._image_logit(fused_self, self._pool(fused_self, weights))
                           + self.image_offset)
        return patch_logits, image_logit, self.type_head(pooled)

    def _fuse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))  # [T, P, h]

    @staticmethod
    def _pool(fused: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (weights[:, None] * fused.flatten(0, 1)).sum(dim=0)

    def _image_logit(self, fused: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        return self.image_head(torch.cat([pooled, fused.flatten(0, 1).amax(dim=0)])).squeeze(-1)


def detector_loss(
    patch_logits: torch.Tensor,
    image_logit: torch.Tensor,
    type_logits: torch.Tensor,
    patch_target: torch.Tensor,
    is_regression: torch.Tensor,
    op: torch.Tensor,
    patch_pos_weight: float = 5.0,
    image_pos_weight: float = 1.0,
    type_weights: torch.Tensor | None = None,
    image_weight: float = 1.0,
    type_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of patch BCE, image BCE and (regressions only) type CE."""

    patch_loss = F.binary_cross_entropy_with_logits(
        patch_logits.float(),
        patch_target,
        pos_weight=patch_logits.new_tensor(patch_pos_weight, dtype=torch.float),
    )
    image_loss = F.binary_cross_entropy_with_logits(
        image_logit.float(),
        is_regression,
        pos_weight=image_logit.new_tensor(image_pos_weight, dtype=torch.float),
    )
    total = patch_loss + image_weight * image_loss
    parts = {"patch": patch_loss.item(), "image": image_loss.item()}
    if op.item() >= 0:
        type_loss = F.cross_entropy(type_logits.float()[None], op[None], weight=type_weights)
        total = total + type_weight * type_loss
        parts["type"] = type_loss.item()
    return total, parts
