"""Frozen DINO full-page distance baseline; no ground-truth-guided windows."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from time import perf_counter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", choices=["dinov2", "dinov3"], required=True)
    p.add_argument("--subset", choices=["train", "val"], default="val")
    p.add_argument("--limit-pages", type=int, help="Sorted first N pages; smoke test only")
    p.add_argument("--limit-pairs", type=int, help="First N sorted pairs; smoke test only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--system-trust", action="store_true")
    a = p.parse_args()
    if a.limit_pages is not None and a.limit_pages < 1:
        p.error("--limit-pages must be positive")
    if a.limit_pairs is not None and a.limit_pairs < 1:
        p.error("--limit-pairs must be positive")
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Output directory must be empty; preserve previous runs.")
    if a.system_trust:
        import truststore
        truststore.inject_into_ssl()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from regression_lens.data.records import load_records
    from regression_lens.data.splits import TEAM_SPLIT, load_split
    from regression_lens.evaluation.tiling import window_origins, add_patch_map, summarize_map

    repo = Path(__file__).resolve().parents[1]
    records = load_records(a.data_root)
    split_path = repo / TEAM_SPLIT
    split = load_split(split_path, records)
    pages = sorted(split[a.subset])
    if a.limit_pages:
        pages = pages[:a.limit_pages]
    selected = sorted((r for r in records if r["parent_id"] in set(pages)),
                      key=lambda r: (r["parent_id"], r["sample_id"]))
    if a.limit_pairs:
        selected = selected[:a.limit_pairs]
    ids = {
        "dinov2": ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
        "dinov3": ("facebook/dinov3-vitb16-pretrain-lvd1689m", "5931719e67bbdb9737e363e781fb0c67687896bc"),
    }
    model_id, revision = ids[a.model]
    device = torch.device(a.device)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in selected environment")
    torch.set_num_threads(4)
    cache = repo / "outputs/dino_fit/model_cache"
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision,
        cache_dir=cache, token=a.model == "dinov3", use_fast=a.model == "dinov3")
    model = AutoModel.from_pretrained(model_id, revision=revision, cache_dir=cache,
        token=a.model == "dinov3", use_safetensors=True,
        attn_implementation="sdpa").to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    patch = model.config.patch_size
    registers = getattr(model.config, "num_register_tokens", 0)
    size, stride, scale = 448, 224, 2
    assert size % patch == 0 and stride % patch == 0
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "maps").mkdir()
    config = dict(model=model_id, revision=revision, subset=a.subset,
        pages=sorted({r["parent_id"] for r in selected}),
        sample_ids=[r["sample_id"] for r in selected], smoke_test=a.limit_pages is not None or a.limit_pairs is not None,
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256((a.data_root/"master_dataset_with_benign.json").read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        window=size, stride=stride, output_scale=scale, patch=patch, registers=registers,
        aggregation="mean of overlapping patch distances on 2px grid",
        score="top5 percent mean on union image area", dtype="float32", attention="sdpa",
        resize=False, center_crop=False, normalization_mean=processor.image_mean,
        normalization_std=processor.image_std, seed=490, trainable_parameters=0,
        padding="white; include union of real reference/candidate image area",
        device=str(device), gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
        python=sys.version, platform=platform.platform(),
        packages={k: importlib.metadata.version(k) for k in ("torch", "transformers", "torchvision", "numpy", "Pillow")})
    (a.out/"config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    torch.manual_seed(490)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    began = perf_counter()
    with (a.out/"predictions.jsonl").open("w") as predictions, (a.out/"pixel_predictions.jsonl").open("w") as pixel_predictions, (a.out/"statistics.jsonl").open("w") as statistics:
        for index, record in enumerate(selected):
            started = perf_counter()
            images = []
            for key in ("reference_path", "candidate_path"):
                with Image.open(a.data_root/record[key]) as image:
                    images.append(image.convert("RGB"))
            width, height = max(im.width for im in images), max(im.height for im in images)
            shape = (math.ceil(height/scale), math.ceil(width/scale))
            totals = {k: np.zeros(shape, np.float32) for k in ("cosine", "l2")}
            counts = np.zeros(shape, np.uint16)
            valid = np.zeros(shape, bool)
            # OR rather than intersection: size changes remain observable.
            for im in images:
                valid[:math.ceil(im.height/scale), :math.ceil(im.width/scale)] = True
            origins = window_origins(width, height, size, stride)
            for x, y in origins:
                crops = []
                for im in images:
                    canvas = Image.new("RGB", (size, size), "white")
                    right, bottom = min(x+size, im.width), min(y+size, im.height)
                    if right > x and bottom > y:
                        canvas.paste(im.crop((x, y, right, bottom)), (0, 0))
                    crops.append(canvas)
                pixels = processor(images=crops, return_tensors="pt", do_resize=False,
                    do_center_crop=False)["pixel_values"].to(device)
                with torch.inference_mode():
                    tokens = model(pixel_values=pixels).last_hidden_state[:, 1+registers:]
                    assert tokens.shape[1] == (size//patch)**2
                    cosine = (1-F.cosine_similarity(tokens[0], tokens[1], dim=-1)).clamp(0, 2)
                    l2 = torch.linalg.vector_norm(tokens[0]-tokens[1], dim=-1)
                    maps = {"cosine": cosine.reshape(size//patch, -1).cpu().numpy(),
                            "l2": l2.reshape(size//patch, -1).cpu().numpy()}
                # Same overlap count for both metrics.
                old_count = counts.copy()
                for key in maps:
                    counts[:] = old_count
                    add_patch_map(totals[key], counts, maps[key], x, y, patch, scale)
            assert np.all(counts[valid] > 0)
            maps = {k: v/np.maximum(counts, 1) for k, v in totals.items()}
            summaries = {k: summarize_map(v, valid, scale) for k, v in maps.items()}
            # Suppress an arbitrary argmax for numerical identity maps.
            point = summaries["cosine"]["point"] if summaries["cosine"]["max"] > 1e-6 else None
            if point:
                point = [min(point[0], width-.5), min(point[1], height-.5)]
            prediction = dict(sample_id=record["sample_id"], score=summaries["cosine"]["top5_mean"], point=point)
            predictions.write(json.dumps(prediction)+"\n")
            predictions.flush()
            # Independent pixel baseline, same white padding and union geometry.
            canvases = []
            for im in images:
                canvas = Image.new("RGB", (width, height), "white")
                canvas.paste(im, (0, 0))
                canvases.append(np.asarray(canvas, dtype=np.float32))
            pixel = np.abs(canvases[0]-canvases[1]).mean(axis=2)/255
            pixel_valid = np.zeros(pixel.shape, bool)
            for im in images:
                pixel_valid[:im.height, :im.width] = True
            pixel_summary = summarize_map(pixel, pixel_valid, scale=1)
            pixel_predictions.write(json.dumps(dict(sample_id=record["sample_id"],
                score=pixel_summary["top5_mean"], point=pixel_summary["point"] if pixel_summary["max"] > 0 else None))+"\n")
            # Record raw maps; plot later without repeating inference.
            np.savez_compressed(a.out/"maps"/f"{index:05d}.npz", **maps, valid=valid,
                                pixel=pixel[::2, ::2])
            statistics.write(json.dumps(dict(sample_id=record["sample_id"], map_file=f"maps/{index:05d}.npz",
                width=width, height=height, windows=len(origins), distances=summaries,
                pixel=pixel_summary, elapsed_seconds=perf_counter()-started))+"\n")
            statistics.flush()
            print(f"{index+1}/{len(selected)} {record['sample_id']} | {len(origins)} windows | {perf_counter()-started:.2f}s", flush=True)
    elapsed = perf_counter()-began
    (a.out/"complete.json").write_text(json.dumps(dict(pairs=len(selected), elapsed_seconds=elapsed,
        pairs_per_second=len(selected)/elapsed, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if a.device == "cuda" else None), indent=2)+"\n")


if __name__ == "__main__":
    main()
