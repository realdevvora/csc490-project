"""Repeat the frozen notebook pilot without executing or modifying the notebook.

Reuses its tested crop and distance functions via AST; no training or new sampling.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_functions(path, namespace):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    wanted = {"load_crop", "distance_maps", "roi_patch_overlap", "spatial_tokens",
              "weighted_mean", "summarize_pair_maps"}
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            nodes.extend(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name in wanted)
    if {n.name for n in nodes} != wanted or len(nodes) != len(wanted):
        raise RuntimeError("Notebook helper contract changed; review the runner.")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, default=Path("outputs/dino_fit"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--system-trust", action="store_true")
    args = parser.parse_args()
    if args.system_trust:
        import truststore
        truststore.inject_into_ssl()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from huggingface_hub import get_token
    from transformers import AutoImageProcessor, AutoModel
    from regression_lens.data.records import load_records
    from regression_lens.evaluation.metrics import peak_point, point_in_box

    repo = Path(__file__).resolve().parents[1]
    root = args.study_root.resolve()
    manifest_path = root / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_root = root / "dinov2/pilot_13"
    baseline = json.loads((baseline_root / "statistics.json").read_text(encoding="utf-8"))
    if baseline["sample_manifest_sha256"] != digest(manifest_path):
        raise RuntimeError("DINOv2 manifest does not match frozen sample.")
    if manifest["metadata_sha256"] != digest(args.data_root / "master_dataset_with_benign.json"):
        raise RuntimeError("Dataset metadata differs from frozen study.")
    if manifest["split_sha256"] != digest(repo / "data/splits/splits_85_15.json"):
        raise RuntimeError("Committed split differs from frozen study.")
    supplement_path = baseline_root / "position_supplement_manifest.json"
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    if supplement["base_manifest_sha256"] != digest(manifest_path):
        raise RuntimeError("Position supplement belongs to a different study.")
    selections = [(t["operator"], t, baseline_root / f"{t['operator']}_native_maps.npz")
                  for t in manifest["trios"]]
    if [t for _, t, _ in selections] != [t["selection"] for t in baseline["trios"]]:
        raise RuntimeError("Baseline selections differ from frozen manifest.")
    selections.append(("position_roi_supplement", supplement["selection"],
                       baseline_root / "position_supplement_native_maps.npz"))
    for _, _, maps_path in selections:
        if not maps_path.is_file():
            raise FileNotFoundError(maps_path)
    if not get_token():
        raise RuntimeError("Authenticate with Hugging Face on this machine before running.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the selected environment.")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    records = {r["sample_id"]: r for r in load_records(args.data_root)}
    notebook = repo / "notebooks/dino_exploration.ipynb"
    helpers = dict(np=np, torch=torch, F=F, math=math, Image=Image,
                   DATA_ROOT=args.data_root, peak_point=peak_point, point_in_box=point_in_box)
    notebook_functions(notebook, helpers)
    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    revision = "5931719e67bbdb9737e363e781fb0c67687896bc"
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision,
        cache_dir=root / "model_cache", token=True, use_fast=True)
    if (list(processor.image_mean) != baseline["normalization_mean"] or
            list(processor.image_std) != baseline["normalization_std"]):
        raise RuntimeError("Normalization differs from baseline.")
    model = AutoModel.from_pretrained(model_id, revision=revision,
        cache_dir=root / "model_cache", token=True, use_safetensors=True,
        attn_implementation="sdpa").to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    patch, size = model.config.patch_size, manifest["crop_size"]
    grid, registers = size // patch, model.config.num_register_tokens
    assert (patch, grid, model.config.hidden_size) == (16, 28, 768)
    destination = root / "dinov3/pilot_13"
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError(f"Preserve existing results before rerunning: {destination}")
    limits = json.loads((baseline_root / "display_limits.json").read_text(encoding="utf-8"))
    labels = ("regression", "benign", "no_change")
    all_rows, trios, comparison = [], [], []
    for name, selection, maps_path in selections:
        chosen = [records[sid] for sid in selection["sample_ids"]]
        regression = chosen[0]
        assert [r["label"] for r in chosen] == list(labels)
        assert all(r["reference_path"] == regression["reference_path"] for r in chosen)
        crop = selection["crop_xyxy"]
        roi = regression["bbox_after"] if regression["reflow"] else regression["diff_bbox"]
        images, validity, features, timings = {}, {}, {}, {}
        paths = {"reference": regression["reference_path"],
                 **{r["label"]: r["candidate_path"] for r in chosen}}
        for label, path in paths.items():
            images[label], validity[label] = helpers["load_crop"](path, crop)
            pixels = processor(images=images[label], return_tensors="pt",
                do_resize=False, do_center_crop=False)["pixel_values"].to(device)
            assert tuple(pixels.shape) == (1, 3, size, size)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            with torch.inference_mode():
                hidden = model(pixel_values=pixels).last_hidden_state
                features[label] = helpers["spatial_tokens"](hidden, grid, registers).cpu()
            timings[label] = perf_counter() - start
        arrays, rows = {}, []
        for record in chosen:
            label = record["label"]
            maps = helpers["distance_maps"](features["reference"], features[label], grid)
            valid = (validity["reference"] & validity[label]).reshape(grid, patch, grid, patch).all(axis=(1, 3))
            arrays.update({f"{label}_{k}": v for k, v in maps.items()})
            arrays[f"{label}_valid_patches"] = valid
            rows.extend(dict(artifact=name, operator=selection["operator"],
                subset="supplement" if name.endswith("supplement") else "baseline", **r)
                for r in helpers["summarize_pair_maps"](record, maps, valid, roi, crop, grid, patch))
        identity = next(r for r in rows if r["label"] == "no_change" and r["distance"] == "l2")
        assert identity["max"] == 0
        np.savez_compressed(destination / f"{name}_native_maps.npz", **arrays)
        figure, axes = plt.subplots(3, 4, figsize=(16, 12), layout="constrained")
        with np.load(maps_path) as saved:
            for row_index, label in enumerate(labels):
                axes[row_index, 0].imshow(images["reference"])
                axes[row_index, 1].imshow(images[label])
                for col, data, step, title in ((2, saved, 14, "DINOv2"), (3, arrays, 16, "DINOv3")):
                    values = data[f"{label}_cosine"]
                    expanded = F.interpolate(torch.from_numpy(values).view(1, 1, *values.shape),
                        size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
                    mask = data[f"{label}_valid_patches"].repeat(step, 0).repeat(step, 1)
                    heat = axes[row_index, col].imshow(np.ma.masked_where(~mask, expanded),
                        cmap="magma", vmin=0, vmax=limits["cosine"]["vmax"])
                    axes[row_index, col].set_title(title)
                axes[row_index, 0].set_title("Reference")
                axes[row_index, 1].set_title(label)
                for col in (1, 2, 3):
                    x, y, w, h = roi
                    axes[row_index, col].add_patch(Rectangle((x-crop[0], y-crop[1]), w, h,
                        fill=False, edgecolor="lime", linestyle="--"))
                for ax in axes[row_index]:
                    ax.axis("off")
        figure.colorbar(heat, ax=axes[:, 2:], shrink=.7, extend="max",
                        label="Raw cosine; frozen DINOv2 display scale")
        figure.suptitle(name)
        figure.savefig(destination / f"{name}_comparison.png", dpi=120)
        plt.close(figure)
        for row in rows:
            previous_rows = (json.loads((baseline_root / "position_supplement_statistics.json").read_text())["statistics"]
                             if name.endswith("supplement") else baseline["statistics"])
            previous = next(r for r in previous_rows if r["sample_id"] == row["sample_id"] and r["distance"] == row["distance"])
            comparison.append(dict(artifact=name, label=row["label"], distance=row["distance"],
                d2_top5=previous["top5_mean"], d3_top5=row["top5_mean"],
                d2_pointing=previous["pointing_hit"], d3_pointing=row["pointing_hit"]))
        all_rows.extend(rows)
        trios.append(dict(artifact=name, selection=selection, forward_seconds=timings))
        print(f"Completed {name} ({len(trios)}/{len(selections)})", flush=True)
    report = dict(model_id=model_id, revision=revision, device=str(device), dtype="float32",
        python=sys.version, platform=platform.platform(), torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        sample_manifest_sha256=digest(manifest_path), notebook_sha256=digest(notebook),
        runner_sha256=digest(Path(__file__)), supplement_manifest_sha256=digest(supplement_path),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        patch_size=patch, grid_size=grid, register_tokens=registers,
        normalization_mean=processor.image_mean, normalization_std=processor.image_std,
        resize=False, center_crop=False, trainable_parameters=0, trios=trios, statistics=all_rows)
    (destination / "statistics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for filename, rows in (("statistics.csv", all_rows), ("comparison.csv", comparison)):
        with (destination / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {len(all_rows)} distance summaries: {destination}")


if __name__ == "__main__":
    main()
