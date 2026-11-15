"""Shared pinned backbone path for the learned-head experiments."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = {
    'dinov2': ('facebook/dinov2-base', 'f9e44c814b77203eaa57a6bdbbd535f21ede1415'),
    'dinov3': ('facebook/dinov3-vitb16-pretrain-lvd1689m', '5931719e67bbdb9737e363e781fb0c67687896bc'),
}
REPO = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Preserve nonempty output before rerunning: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance():
    return dict(python=sys.version, platform=platform.platform(),
        packages={k: importlib.metadata.version(k) for k in ('torch', 'torchvision', 'transformers', 'numpy', 'Pillow')},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()),
        sources={str(p.relative_to(REPO)): sha(p) for p in
                 [Path(__file__), REPO/'src/regression_lens/models/dino_head.py', REPO/'scripts/dino_heads.py']})


class Backbone:
    def __init__(self, name, device):
        from transformers import AutoImageProcessor, AutoModel
        self.device = torch.device(device)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        model_id, revision = MODELS[name]
        options = dict(revision=revision, cache_dir=REPO/'outputs/dino_fit/model_cache', token=name == 'dinov3')
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=name == 'dinov3', **options)
        self.model = AutoModel.from_pretrained(model_id, use_safetensors=True,
            attn_implementation='sdpa', **options).to(device=self.device, dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self.patch = self.model.config.patch_size
        self.grid = 448//self.patch
        self.registers = getattr(self.model.config, 'num_register_tokens', 0)

    @torch.no_grad()
    def features(self, images):
        inputs = self.processor(images=images, return_tensors='pt', do_resize=False,
                                do_center_crop=False)['pixel_values'].to(self.device)
        assert tuple(inputs.shape[1:]) == (3, 448, 448)
        hidden = self.model(pixel_values=inputs).last_hidden_state
        assert hidden.shape[1] == 1+self.registers+self.grid**2
        return hidden[:, 1+self.registers:].transpose(1, 2).reshape(-1, 768, self.grid, self.grid)

    def config(self):
        return dict(patch=self.patch, grid=self.grid, registers=self.registers, dtype='float32',
            resize=False, center_crop=False, attention='sdpa',
            mean=list(self.processor.image_mean), std=list(self.processor.image_std))


def crop_image(image, crop):
    x, y, right, bottom = crop
    canvas = Image.new('RGB', (right-x, bottom-y), 'white')
    right, bottom = min(right, image.width), min(bottom, image.height)
    if right > x and bottom > y:
        canvas.paste(image.crop((x, y, right, bottom)), (0, 0))
    return canvas


def open_pair(root, record):
    images = []
    for key in ('reference_path', 'candidate_path'):
        with Image.open(Path(root)/record[key]) as image:
            images.append(image.convert('RGB'))
    return images
