"""Train and evaluate a Siamese UIClip (or CLIP/SigLIP) regression detector.

Examples:
  # Smoke test on a few pages
  python scripts/train_uiclip.py --exp smoke --limit-pages 10 --epochs 1

  # Stage 1: frozen encoder, train heads only
  python scripts/train_uiclip.py --exp uiclip_frozen

  # Same model, much faster after the first run: reuse cached encoder features
  python scripts/train_uiclip.py --exp uiclip_frozen --cache-features

  # Stage 2: also fine-tune the last 4 encoder blocks
  python scripts/train_uiclip.py --exp uiclip_ft4 --train-last-n 4

  # Ablation: the model before self-calibration (identical pairs can be flagged)
  python scripts/train_uiclip.py --exp uiclip_frozen_nocal --no-self-calibration

  # Ablation: no browser-render benign pairs in training
  python scripts/train_uiclip.py --exp uiclip_frozen_nobenign --exclude-benign

  # Backbone comparisons
  python scripts/train_uiclip.py --exp clip_frozen --checkpoint openai/clip-vit-base-patch32
  python scripts/train_uiclip.py --exp siglip_frozen --checkpoint google/siglip-base-patch16-512

  # Re-evaluate a saved run's checkpoints
  python scripts/train_uiclip.py --exp uiclip_frozen --eval-only

Each run writes outputs/<exp>/{config.json, metrics.json, notes.md} and keeps
two checkpoints, the epoch with the best validation AUROC (regression vs
benign) and the epoch with the best validation F1:

  outputs/<exp>/best_auroc/{model.pt, val_predictions.jsonl}
  outputs/<exp>/best_f1/{model.pt, val_predictions.jsonl}

val_predictions.jsonl follows docs/prediction-format.md, so
scripts/score_predictions.py scores it the same way as any other model. The
decision threshold and best epoch are both chosen on validation, so treat
validation numbers as optimistic until there is a held-out test split.

--cache-features stores the output of the frozen encoder blocks once per
screenshot (under --cache-dir; about 53 GB for UIClip/CLIP, ~285 GB for SigLIP)
and trains from it, several times faster. Features are bit-identical to
recomputing them, except bf16 rounding in white padding tiles (pairs whose
candidate is taller than the reference). The cache is specific to the backbone
and --train-last-n, and is built (or completed) on first use.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from regression_lens.data.feature_cache import FeatureCache, build_feature_cache
from regression_lens.data.pairs import PairDataset, first_item, load_records, operator_names, patches_to_cells
from regression_lens.data.splits import load_or_create_split
from regression_lens.evaluation.metrics import largest_component_box, peak_point
from regression_lens.evaluation.scoring import FORMAT_VERSION, score_predictions, write_predictions
from regression_lens.models.uiclip import UICLIP_CHECKPOINT, SiameseDetector, detector_loss, encoder_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, help="run name under outputs/")
    parser.add_argument("--data-root", default="d2c_v2")
    parser.add_argument("--split", default="data/processed/splits_placeholder_85_15.json",
                        help="page split JSON; created (seeded 85/15) if it does not exist")
    parser.add_argument("--checkpoint", default=UICLIP_CHECKPOINT)
    parser.add_argument("--train-last-n", type=int, default=0, help="encoder blocks to fine-tune")
    parser.add_argument("--no-self-calibration", action="store_true",
                        help="old model: do not calibrate scores against the reference vs itself")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--accum", type=int, default=8, help="pages per optimizer step")
    parser.add_argument("--patch-pos-weight", type=float, default=5.0)
    parser.add_argument("--map-threshold", type=float, default=0.5,
                        help="change-map probability for the localization box")
    parser.add_argument("--exclude-benign", action="store_true",
                        help="train without browser-render benign pairs (still evaluated)")
    parser.add_argument("--limit-pages", type=int, default=None, help="use only N train/val pages")
    parser.add_argument("--cache-features", action="store_true",
                        help="train from cached frozen-encoder features (built on first use)")
    parser.add_argument("--cache-dir", default="data/processed/features")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true",
                        help="re-evaluate outputs/<exp>/best_*/model.pt without training")
    return parser.parse_args()


# Checkpoint selections: each keeps the epoch that maximizes its validation metric.
SELECTIONS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "best_auroc": lambda m: m["auroc"]["regression_vs_benign"],
    "best_f1": lambda m: m["detection"]["f1"],
}


def summary_line(metrics: dict[str, Any]) -> str:
    def fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    return (f"AUROC {fmt(metrics['auroc']['regression_vs_benign'])}  F1 {fmt(metrics['detection']['f1'])}  "
            f"FPR {metrics['false_positive_rate']}  "
            f"type macro-F1 {fmt(metrics.get('type', {}).get('macro_f1'))}  "
            f"pointing {fmt(metrics.get('localization', {}).get('pointing_accuracy'))}")


def git_state() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def to_device(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(item)
    for key in ("ref", "cand", "patch_target"):
        moved[key] = item[key].to(device, non_blocking=True)
    moved["is_regression"] = torch.tensor(item["is_regression"], device=device)
    moved["op"] = torch.tensor(item["op"], device=device)
    return moved


class LossMeter:
    """Mean of each loss part over the pairs that have it (type: regressions only)."""

    def __init__(self) -> None:
        self.sums: Counter[str] = Counter()
        self.counts: Counter[str] = Counter()

    def update(self, parts: dict[str, float]) -> None:
        self.sums.update(parts)
        self.counts.update(parts.keys())

    def means(self) -> dict[str, float]:
        return {key: self.sums[key] / self.counts[key] for key in self.sums}

    def __str__(self) -> str:
        return " ".join(f"{key} {value:.4f}" for key, value in self.means().items())


def pair_loss(outputs: tuple[torch.Tensor, ...], item: dict[str, Any],
              loss_kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    loss, parts = detector_loss(*outputs, item["patch_target"], item["is_regression"], item["op"],
                                **loss_kwargs)
    return loss, {"total": loss.item(), **parts}


@torch.no_grad()
def predict(model: SiameseDetector, loader: DataLoader, device: torch.device, patch: int,
            map_threshold: float, operators: list[str], model_name: str,
            loss_kwargs: dict[str, Any]) -> tuple[list[dict[str, Any]], LossMeter]:
    """Predict every pair in the format of docs/prediction-format.md, and its loss."""

    model.eval()
    predictions, losses = [], LossMeter()
    for item in loader:
        sample_id = item["record"]["sample_id"]
        item = to_device(item, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(item["ref"], item["cand"], item["cached"])
        patch_logits, image_logit, type_logits = outputs
        losses.update(pair_loss(outputs, item, loss_kwargs)[1])
        change_map = patches_to_cells(torch.sigmoid(patch_logits.float()), item["grid"]).cpu().numpy()
        cell_px = patch / item["scale"]  # one change-map cell in original pixels
        predictions.append({
            "sample_id": sample_id,
            "score": torch.sigmoid(image_logit.float()).item(),
            "op_pred": operators[int(type_logits.argmax())],
            "box": largest_component_box(change_map, map_threshold, cell_px),
            "point": list(peak_point(change_map, cell_px)),
            "model": model_name,
            "format_version": FORMAT_VERSION,
        })
    return predictions, losses


# Settings that define the model architecture; --eval-only restores them from
# the run's config.json so a checkpoint is always rebuilt as it was trained.
MODEL_SETTINGS = {"checkpoint": None, "train_last_n": None,
                  "no_self_calibration": True}  # runs from before the option were uncalibrated


def main() -> None:
    args = parse_args()
    saved_config = Path("outputs") / args.exp / "config.json"
    if args.eval_only and saved_config.exists():
        saved = json.loads(saved_config.read_text())
        for key, missing in MODEL_SETTINGS.items():
            value = saved.get(key, missing)
            if value is not None and value != getattr(args, key):
                print(f"eval-only: using {key}={value!r} from {saved_config}")
                setattr(args, key, value)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("outputs") / args.exp
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data_root)
    split = load_or_create_split(args.split, records)
    train_pages, val_pages = split["train"], split["val"]
    if args.limit_pages:
        train_pages, val_pages = train_pages[: args.limit_pages], val_pages[: args.limit_pages]
    operators = operator_names(records)

    info = encoder_info(args.checkpoint)
    dataset_kwargs = dict(records=records, data_root=args.data_root, operators=operators,
                          tile=info.tile, patch=info.patch, mean=info.mean, std=info.std)
    train_labels = ("regression", "no_change") if args.exclude_benign else ("regression", "no_change", "benign")
    train_set = PairDataset(pages=train_pages, labels=train_labels, **dataset_kwargs)
    val_set = PairDataset(pages=val_pages, **dataset_kwargs)
    print(f"train pairs {len(train_set)}  val pairs {len(val_set)}  tile {info.tile}  patch {info.patch}")

    model = SiameseDetector(args.checkpoint, len(operators), args.train_last_n,
                            self_calibrate=not args.no_self_calibration).to(device)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}

    cache = None
    if args.cache_features:
        cache = FeatureCache.open(args.cache_dir, args.checkpoint, model.cache_layer,
                                  info.tile, info.patch, train_set.cols, info.mean, info.std)
        build_feature_cache(cache, model, {**train_set.images(), **val_set.images()},
                            args.data_root, device, args.workers)
        train_set.feature_cache = val_set.feature_cache = cache
        print(f"using feature cache {cache.directory}")

    loader_kwargs = dict(batch_size=1, collate_fn=first_item, num_workers=args.workers,
                         pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    config = {**vars(args), "operators": operators, "encoder": vars(info), "git": git_state(),
              "torch": torch.__version__,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
              "n_train_pairs": len(train_set), "n_val_pairs": len(val_set),
              "feature_cache": str(cache.directory) if cache else None}
    if not args.eval_only:
        (out / "config.json").write_text(json.dumps(config, indent=2))

    # Rebalance the 4:1 regression:negative ratio and the rare operators.
    labels = Counter(r["label"] for r in train_set.records)
    image_pos_weight = (labels["no_change"] + labels["benign"]) / max(labels["regression"], 1)
    op_counts = Counter(r["operator"] for r in train_set.records if r["operator"])
    type_weights = torch.tensor([1 / max(op_counts[o], 1) for o in operators], device=device)
    type_weights = type_weights / type_weights.mean()
    loss_kwargs = dict(patch_pos_weight=args.patch_pos_weight, image_pos_weight=image_pos_weight,
                       type_weights=type_weights)

    def evaluate(epoch: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        predictions, val_loss = predict(model, val_loader, device, info.patch, args.map_threshold,
                                        operators, args.exp, loss_kwargs)
        metrics = {"epoch": epoch, "val_loss": val_loss.means(), **score_predictions(
            predictions, val_set.records, operators,
            expected_ids=[r["sample_id"] for r in val_set.records])}
        return metrics, predictions

    if args.eval_only:
        checkpoints = [out / name / "model.pt" for name in SELECTIONS if (out / name / "model.pt").exists()]
        if not checkpoints:
            raise SystemExit(f"no checkpoints under {out}/best_*/")
        for checkpoint in checkpoints:
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
            metrics, predictions = evaluate(None)
            write_predictions(checkpoint.parent / "val_predictions.jsonl", predictions)
            (checkpoint.parent / "metrics_eval_only.json").write_text(json.dumps(metrics, indent=2))
            print(f"{checkpoint.parent.name}: {summary_line(metrics)}")
        return

    optimizer = torch.optim.AdamW(
        [{"params": model.head_parameters(), "lr": args.lr_head},
         {"params": model.encoder_parameters(), "lr": args.lr_encoder}],
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best = {name: -math.inf for name in SELECTIONS}
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss, running = LossMeter(), LossMeter()
        for step, item in enumerate(train_loader, start=1):
            item = to_device(item, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(item["ref"], item["cand"], item["cached"])
            loss, parts = pair_loss(outputs, item, loss_kwargs)
            (loss / args.accum).backward()
            if step % args.accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss.update(parts)
            running.update(parts)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} {running}")
                running = LossMeter()

        metrics, predictions = evaluate(epoch)
        metrics = {"epoch": epoch, "train_loss": epoch_loss.means(), **metrics}
        history.append(metrics)
        print(f"epoch {epoch} loss  train: {epoch_loss}")
        print(f"epoch {epoch} loss  val:   " + " ".join(f"{k} {v:.4f}" for k, v in metrics["val_loss"].items()))
        print(f"epoch {epoch} val {summary_line(metrics)}")

        state = None
        for name, select in SELECTIONS.items():
            value = select(metrics)
            if value is not None and value > best[name]:
                best[name] = value
                state = state or {k: v for k, v in model.state_dict().items() if k in trainable}
                (out / name).mkdir(exist_ok=True)
                torch.save(state, out / name / "model.pt")
                write_predictions(out / name / "val_predictions.jsonl", predictions)
                print(f"epoch {epoch} new {name} ({value:.4f})")
        best_epoch = {
            name: max((m for m in history if select(m) is not None), key=select, default={"epoch": None})["epoch"]
            for name, select in SELECTIONS.items()
        }
        (out / "metrics.json").write_text(json.dumps({"best_epoch": best_epoch, "history": history}, indent=2))

    notes = out / "notes.md"
    if not notes.exists():
        notes.write_text(f"# {args.exp}\n\nObservations, failures and deviations from the protocol:\n\n- \n")


if __name__ == "__main__":
    main()
