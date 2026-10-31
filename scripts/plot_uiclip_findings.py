"""Plot the UIClip/CLIP experiment findings as slide-ready PNG figures.

Reads the 10-epoch runs in outputs/ (metrics.json, best_auroc predictions),
the scorer's per-run results in outputs/comparison/ and the batch logs in
outputs/logs/, and writes figures to two folders under docs/figures/uiclip/:

  first_batch/      results of the first batch only (scripts/run_uiclip_experiments.sh)
  all_experiments/  results of both batches, including CLIP fine-tuning and SigLIP
                    (scripts/run_clip_siglip_experiments.sh)

  python scripts/plot_uiclip_findings.py                        # both folders
  python scripts/plot_uiclip_findings.py --only all_experiments # just one

Requires matplotlib (``pip install -e ".[figures]"``).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import textwrap
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402
from PIL import Image  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-3, which validate as
# colorblind-safe for every pair, plus chart chrome and text ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#1c5cab"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"

EXPERIMENTS = {  # name -> (display label, backbone, encoder, benign pairs, calibration)
    "uiclip_cal": ("UIClip, frozen", "UIClip", "Frozen", "Yes", "On"),
    "uiclip_nocal": ("UIClip, frozen, no calibration", "UIClip", "Frozen", "Yes", "Off"),
    "clip_cal": ("CLIP, frozen", "CLIP", "Frozen", "Yes", "On"),
    "uiclip_cal_nobenign": ("UIClip, frozen, no benign pairs", "UIClip", "Frozen", "No", "On"),
    "uiclip_cal_ft4": ("UIClip, last 4 blocks fine-tuned", "UIClip", "Last 4 of 12 blocks trained", "Yes", "On"),
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.labelcolor": INK_2,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def title(fig: plt.Figure, heading: str, subtitle: str, below_pt: float = 18) -> float:
    """Draw the heading and wrapped subtitle; return the figure fraction left for plots.

    ``below_pt`` is the space to leave under the subtitle (e.g. for panel titles).
    """

    width_in, height_in = fig.get_size_inches()
    lines = textwrap.wrap(subtitle, width=int(width_in * 12.5))
    fig.text(0.02, 0.985, heading, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.985, "\n".join(lines), ha="left", va="top", fontsize=10.5, color=INK_2, linespacing=1.35,
             transform=offset_copy(fig.transFigure, fig=fig, y=-24, units="points"))
    used_pt = 0.015 * height_in * 72 + 24 + 14.5 * len(lines) + below_pt
    return 1 - used_pt / (height_in * 72)


def load_results(root: Path) -> dict[str, list[dict]]:
    """Scorer metrics of each experiment's best-AUROC checkpoints, one per seed."""

    results = defaultdict(list)
    for path in sorted((root / "comparison").glob("*_10_epochs_s*__best_auroc.metrics.json")):
        exp = re.sub(r"_10_epochs_s\d+__best_auroc\.metrics\.json$", "", path.name)
        results[exp].append(json.loads(path.read_text()))
    return results


def hbar(ax: plt.Axes, y: float, value: float, height: float, color: str, radius_px: float = 4) -> None:
    """Horizontal bar from x=0 with a 4 px rounded data end and a square baseline end."""

    if value <= 0:
        return
    ax.figure.canvas.draw()
    box = ax.get_window_extent()
    rx = min(radius_px * (ax.get_xlim()[1] - ax.get_xlim()[0]) / box.width, value / 2)
    ry = min(radius_px * (ax.get_ylim()[1] - ax.get_ylim()[0]) / box.height, height / 2)
    y0, y1 = y - height / 2, y + height / 2
    arc = np.linspace(0, np.pi / 2, 12)
    lower = [(value - rx + rx * np.sin(t), y0 + ry - ry * np.cos(t)) for t in arc]
    upper = [(value - rx + rx * np.cos(t), y1 - ry + ry * np.sin(t)) for t in arc]
    ax.add_patch(Polygon([(0, y0), *lower, *upper, (0, y1)], closed=True, facecolor=color, edgecolor="none"))


def fig_experiments(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    top = title(fig, "Five experiments, each trained 3 times",
                "10 epochs per run, seeds 0, 1 and 2; validated on 73 pages (1,108 pairs). "
                "Each row changes one setting from the first row.")
    fig.subplots_adjust(top=top, bottom=0.1, left=0.02, right=0.98)
    ax.axis("off")
    columns = ["Experiment", "Backbone", "Encoder", "Benign pairs\nin training", "Self-\ncalibration"]
    rows = [list(v) for v in EXPERIMENTS.values()]
    baseline = rows[0]
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left", colLoc="left",
                     colWidths=[0.33, 0.11, 0.26, 0.16, 0.14], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1)
        cell.set_facecolor(SURFACE)
        cell.PAD = 0.04
        text = cell.get_text()
        text.set_fontsize(10.5)
        if r == 0:
            text.set_color(INK_2)
            text.set_fontweight("bold")
        else:
            changed = c > 0 and rows[r - 1][c] != baseline[c]
            text.set_color(INK)
            if changed:  # the one setting this experiment varies
                text.set_fontweight("bold")
                cell.set_facecolor("#eef4fc")
    fig.text(0.02, 0.03, "Highlighted cell: the setting that differs from the first row.", fontsize=9.5, color=MUTED)
    fig.savefig(out / "1_experiments.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_benign(results: dict[str, list[dict]], out: Path) -> None:
    groups = [("With benign pairs", results["uiclip_cal"], BLUE), ("Without benign pairs", results["uiclip_cal_nobenign"], ORANGE)]
    panels = [("Benign pairs flagged as regressions", lambda m: m["false_positive_rate"]["benign"]),
              ("Regressions caught (recall)", lambda m: m["detection"]["recall"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    top = title(fig, "Without benign training pairs, every browser difference is flagged",
                "UIClip, frozen encoder. Bars: mean of 3 seeds; dots: individual seeds. "
                "Benign = the same page rendered in Firefox or WebKit.", below_pt=30)
    for ax, (heading, get) in zip(axes, panels):
        ax.set_title(heading, pad=10)
        ax.set_xlim(0, 1.15)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for i, (label, runs, color) in enumerate(groups):
            values = [get(m) for m in runs]
            mean = statistics.mean(values)
            y = 1 - i
            hbar(ax, y, mean, 0.42, color)
            ax.scatter(values, [y] * len(values), s=22, color=INK, zorder=3, linewidths=0)
            ax.text(max(mean, *values) + 0.03, y, f"{mean:.0%}", va="center", ha="left", fontsize=11,
                    color=INK, fontweight="bold")
        ax.set_yticks([1, 0], [g[0] for g in groups])
    fig.subplots_adjust(top=top, wspace=0.12, left=0.19, right=0.98, bottom=0.1)
    fig.savefig(out / "2_benign_training.png", dpi=200)
    plt.close(fig)


def fig_metrics(results: dict[str, list[dict]], out: Path) -> None:
    order = ["uiclip_cal_ft4", "uiclip_cal", "uiclip_nocal", "clip_cal"]
    panels = [("AUROC (reg. vs benign)", lambda m: m["auroc"]["regression_vs_benign"], (0.9875, 1.001), [0.99, 0.995, 1.0], "{:.3f}"),
              ("F1", lambda m: m["detection"]["f1"], (0.969, 1.001), [0.97, 0.98, 0.99, 1.0], "{:.3f}"),
              ("Type macro-F1", lambda m: m["type"]["macro_f1"], (0.24, 0.51), [0.3, 0.4, 0.5], "{:.2f}"),
              ("Pointing accuracy", lambda m: m["localization"]["pointing_accuracy"], (0.33, 0.52), [0.35, 0.4, 0.45, 0.5], "{:.2f}")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    top = title(fig, "Fine-tuning the last 4 blocks gives the best detection and type accuracy; UIClip edges out CLIP",
                "Large dot: mean of 3 seeds (value shown); small dots: individual seeds. Higher is better. Each axis is "
                "zoomed to its metric's range. The no-benign run is omitted (see the benign-pairs figure).", below_pt=30)
    labels = [EXPERIMENTS[e][0] for e in order]
    for ax, (heading, get, limits, ticks, fmt) in zip(axes, panels):
        ax.set_title(heading, pad=10, fontsize=11)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for i, exp in enumerate(order):
            values = [get(m) for m in results[exp]]
            y = len(order) - 1 - i
            ax.scatter(values, [y] * len(values), s=26, color=BLUE_LIGHT, zorder=2, linewidths=0)
            mean = statistics.mean(values)
            ax.scatter([mean], [y], s=80, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=2)
            ax.text(mean, y + 0.28, fmt.format(mean), ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_ylim(-0.6, len(order) - 0.3)
        ax.set_yticks(range(len(order) - 1, -1, -1), labels)
    fig.subplots_adjust(top=top, wspace=0.12, left=0.215, right=0.99, bottom=0.1)
    fig.savefig(out / "3_model_comparison.png", dpi=200)
    plt.close(fig)


def fig_identical(root: Path, data_root: Path, out: Path, seed: int) -> None:
    records = {r["sample_id"]: r for r in json.loads((data_root / "master_dataset_with_benign.json").read_text())}

    def truly_identical(r: dict) -> bool:
        a = np.asarray(Image.open(data_root / r["reference_path"]).convert("RGB"))
        b = np.asarray(Image.open(data_root / r["candidate_path"]).convert("RGB"))
        return a.shape == b.shape and bool((a == b).all())

    rows = [("Without self-calibration", "uiclip_nocal"), ("With self-calibration", "uiclip_cal")]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    top = title(fig, "Self-calibration gives every identical screenshot pair the same score",
                f"Regression score of each validation 'no-change' pair (UIClip frozen, seed {seed}). "
                "Pairs to the right of a row's threshold mark are flagged as regressions.", below_pt=12)
    identity: dict[str, bool] = {}
    rng = np.random.default_rng(0)
    for i, (label, exp) in enumerate(rows):
        run = root / f"{exp}_10_epochs_s{seed}"
        threshold = json.loads((root / "comparison" / f"{run.name}__best_auroc.metrics.json").read_text())["threshold"]
        preds = [json.loads(line) for line in open(run / "best_auroc" / "val_predictions.jsonl")]
        preds = [p for p in preds if records[p["sample_id"]]["label"] == "no_change"]
        for p in preds:
            if p["sample_id"] not in identity:
                identity[p["sample_id"]] = truly_identical(records[p["sample_id"]])
        y = len(rows) - 1 - i
        for same, color in ((True, BLUE), (False, ORANGE)):
            xs = [p["score"] for p in preds if identity[p["sample_id"]] == same]
            ax.scatter(xs, y + rng.uniform(-0.13, 0.13, len(xs)), s=40, color=color, alpha=0.85,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.plot([threshold, threshold], [y - 0.3, y + 0.3], color=INK_2, linewidth=2, zorder=2)
        ax.text(threshold, y + 0.34, f"threshold {threshold:.2f}", ha="center", va="bottom", fontsize=9, color=INK_2)
        distinct = len({round(p["score"], 6) for p in preds if identity[p["sample_id"]]})
        n_same = sum(identity[p["sample_id"]] for p in preds)
        ax.text(1.04, y, f"{n_same} identical pairs,\n{distinct} distinct score{'s' if distinct != 1 else ''}",
                va="center", ha="left", fontsize=9.5, color=INK)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_yticks(range(len(rows) - 1, -1, -1), [r[0] for r in rows])
    ax.set_xlabel("Predicted regression score")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=BLUE, label="Pixel-identical pair"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=ORANGE,
                          label="Labelled no-change, but pixels differ (dataset issue)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, fontsize=9.5)
    fig.subplots_adjust(top=top, left=0.2, right=0.83, bottom=0.28)
    fig.savefig(out / "4_identical_pairs.png", dpi=200)
    plt.close(fig)


def fig_types(results: dict[str, list[dict]], out: Path) -> None:
    def per_type(exp: str) -> dict[str, float]:
        scores = defaultdict(list)
        for m in results[exp]:
            for name, value in m["type"]["per_class"].items():
                scores[name].append(value["f1"])
        return {name: statistics.mean(v) for name, v in scores.items()}

    frozen, tuned = per_type("uiclip_cal"), per_type("uiclip_cal_ft4")
    names = sorted(tuned, key=lambda n: tuned[n])
    fig, ax = plt.subplots(figsize=(10, 5.8))
    top = title(fig, "Colour and border changes are easy to name; spacing and font size are not",
                "Mutation-type F1 per CSS operator, mean of 3 seeds. Fine-tuning improves most types, but subtle layout "
                "changes stay hard, and margin gets worse (0.21 to 0.09).")
    for y, name in enumerate(names):
        ax.plot([frozen[name], tuned[name]], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
        ax.scatter([frozen[name]], [y], s=60, color=BLUE_LIGHT, zorder=2, edgecolors=SURFACE, linewidths=2)
        ax.scatter([tuned[name]], [y], s=60, color=BLUE_DARK, zorder=3, edgecolors=SURFACE, linewidths=2)
    for name in (names[0], names[-1]):  # label the fine-tuned extremes, on the side away from the frozen dot
        y, right = names.index(name), tuned[name] >= frozen[name]
        ax.text(tuned[name] + (0.025 if right else -0.025), y, f"{tuned[name]:.2f}", va="center",
                ha="left" if right else "right", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=BLUE_LIGHT, label="Frozen encoder"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=BLUE_DARK, label="Last 4 blocks fine-tuned")]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5)
    fig.subplots_adjust(top=top, left=0.17, right=0.97, bottom=0.1)
    fig.savefig(out / "5_type_f1.png", dpi=200)
    plt.close(fig)


def fig_curves(root: Path, out: Path) -> None:
    series = [("UIClip, frozen", "uiclip_cal", BLUE), ("CLIP, frozen", "clip_cal", ORANGE),
              ("UIClip, last 4 blocks fine-tuned", "uiclip_cal_ft4", AQUA)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    top = title(fig, "Fine-tuning reaches a lower validation loss, then overfits after epoch 6",
                "Validation loss by epoch: line = mean of 3 seeds, band = range across seeds. Each run keeps its "
                "best-AUROC epoch, so the later overfitting is not in the saved models.")
    for label, exp, color in series:
        losses = np.array([[e["val_loss"]["total"] for e in json.loads((root / f"{exp}_10_epochs_s{s}" / "metrics.json").read_text())["history"]]
                           for s in (0, 1, 2)])
        epochs = np.arange(1, losses.shape[1] + 1)
        ax.fill_between(epochs, losses.min(0), losses.max(0), color=color, alpha=0.12, linewidth=0)
        ax.plot(epochs, losses.mean(0), color=color, linewidth=2, label=label, solid_capstyle="round")
        best = int(losses.mean(0).argmin())
        ax.scatter([epochs[best]], [losses.mean(0)[best]], s=60, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.text(0.02, 0.02, "Dot: each series' lowest mean validation loss.", fontsize=9.5, color=MUTED)
    fig.subplots_adjust(top=top, left=0.09, right=0.98, bottom=0.17)
    fig.savefig(out / "6_validation_loss.png", dpi=200)
    plt.close(fig)


# Every experiment of both batches: name -> (label, backbone, encoder, benign pairs, calibration, seeds, batch)
ALL_EXPERIMENTS = {
    "uiclip_cal": ("UIClip, frozen", "UIClip", "Frozen", "Yes", "On", "3", "First"),
    "uiclip_nocal": ("UIClip, frozen, no calibration", "UIClip", "Frozen", "Yes", "Off", "3", "First"),
    "uiclip_cal_nobenign": ("UIClip, frozen, no benign pairs", "UIClip", "Frozen", "No", "On", "3", "First"),
    "clip_cal": ("CLIP, frozen", "CLIP", "Frozen", "Yes", "On", "3", "First"),
    "siglip_cal": ("SigLIP-512, frozen", "SigLIP-512", "Frozen", "Yes", "On", "3", "Follow-up"),
    "uiclip_cal_ft4": ("UIClip, fine-tuned", "UIClip", "Last 4 of 12 blocks trained", "Yes", "On", "3", "First"),
    "clip_cal_ft4": ("CLIP, fine-tuned", "CLIP", "Last 4 of 12 blocks trained", "Yes", "On", "3", "Follow-up"),
    "siglip_cal_ft4": ("SigLIP-512, fine-tuned", "SigLIP-512", "Last 4 of 12 blocks trained", "Yes", "On", "1", "Follow-up"),
}


def fig_all_experiments(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    top = title(fig, "Eight experiments across two batches, 22 training runs",
                "10 epochs per run; validated on 73 pages (1,108 pairs). Highlighted cells: settings that differ from "
                "the first row. SigLIP-512 uses 512 px tiles and 16 px patches; UIClip and CLIP use 224 px tiles and 32 px patches.")
    fig.subplots_adjust(top=top, bottom=0.04, left=0.02, right=0.98)
    ax.axis("off")
    columns = ["Experiment", "Backbone", "Encoder", "Benign pairs\nin training", "Self-\ncalibration", "Seeds", "Batch"]
    rows = [list(v) for v in ALL_EXPERIMENTS.values()]
    baseline = rows[0]
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left", colLoc="left",
                     colWidths=[0.25, 0.11, 0.22, 0.12, 0.11, 0.07, 0.10], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1)
        cell.set_facecolor(SURFACE)
        cell.PAD = 0.04
        text = cell.get_text()
        text.set_fontsize(10.5)
        if r == 0:
            text.set_color(INK_2)
            text.set_fontweight("bold")
            continue
        text.set_color(INK)
        if 1 <= c <= 4 and rows[r - 1][c] != baseline[c]:  # a setting this experiment varies
            text.set_fontweight("bold")
            cell.set_facecolor("#eef4fc")
    fig.savefig(out / "1_experiments.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Follow-up batch (scripts/run_clip_siglip_experiments.sh): every backbone, frozen and fine-tuned.
BACKBONES = [("UIClip", "uiclip_cal", "uiclip_cal_ft4"), ("CLIP", "clip_cal", "clip_cal_ft4"),
             ("SigLIP-512", "siglip_cal", "siglip_cal_ft4")]
# Mutation operators grouped by family, so confusions within a family show as blocks.
FAMILIES = [("Colour", ["background_color", "text_color"]), ("Appearance", ["opacity"]),
            ("Shape", ["border_radius", "border_width"]),
            ("Typography", ["font_size", "font_weight", "letter_spacing", "line_height"]),
            ("Layout", ["margin", "padding", "position", "width"])]
# Runs whose time includes building a feature cache (excluded from the per-run time).
CACHE_BUILDS = {"clip_cal_10_epochs_s0", "uiclip_cal_ft4_10_epochs_s0", "siglip_cal_10_epochs_s0",
                "clip_cal_ft4_10_epochs_s0"}


def fig_backbones(results: dict[str, list[dict]], out: Path) -> None:
    panels = [("AUROC (reg. vs benign)", lambda m: m["auroc"]["regression_vs_benign"], (0.9895, 1.0015), [0.99, 0.995, 1.0], "{:.4f}"),
              ("Benign FPR", lambda m: m["false_positive_rate"]["benign"], (-0.004, 0.052), [0, 0.02, 0.04], "{:.1%}"),
              ("Type macro-F1", lambda m: m["type"]["macro_f1"], (0.29, 0.51), [0.3, 0.4, 0.5], "{:.2f}"),
              ("Pointing accuracy", lambda m: m["localization"]["pointing_accuracy"], (0.37, 0.61), [0.4, 0.5, 0.6], "{:.2f}"),
              ("Mean box IoU", lambda m: m["localization"]["mean_iou"], (0.145, 0.225), [0.15, 0.2], "{:.2f}")]
    fig, axes = plt.subplots(1, len(panels), figsize=(13.5, 3.6), sharey=True)
    top = title(fig, "Fine-tuning matters more than the backbone; finer patches (SigLIP) point better, but boxes don't improve",
                "Mean of 3 seeds per dot, except SigLIP fine-tuned (1 seed). Each row shows a backbone frozen (light) and "
                "with its last 4 blocks fine-tuned (dark). Benign FPR: lower is better; all others: higher is better. "
                "Axes are zoomed to each metric's range.", below_pt=30)
    for ax, (heading, get, limits, ticks, fmt) in zip(axes, panels):
        ax.set_title(heading, pad=10, fontsize=11)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks] if "%" in fmt else None)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for i, (_, frozen, tuned) in enumerate(BACKBONES):
            y = len(BACKBONES) - 1 - i
            a, b = (statistics.mean(get(m) for m in results[e]) for e in (frozen, tuned))
            ax.plot([a, b], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
            ax.scatter([a], [y], s=70, color=BLUE_LIGHT, zorder=2, edgecolors=SURFACE, linewidths=2)
            ax.scatter([b], [y], s=70, color=BLUE_DARK, zorder=3, edgecolors=SURFACE, linewidths=2)
            ax.text(b, y + 0.22, fmt.format(b), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_ylim(-0.6, len(BACKBONES) - 0.35)
        ax.set_yticks(range(len(BACKBONES) - 1, -1, -1), [name for name, _, _ in BACKBONES])
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=BLUE_LIGHT, label="Frozen encoder"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=BLUE_DARK,
                          label="Last 4 blocks fine-tuned (value shown)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=top, wspace=0.18, left=0.09, right=0.99, bottom=0.2)
    fig.savefig(out / "2_backbones_frozen_vs_finetuned.png", dpi=200)
    plt.close(fig)


def fig_curves_by_backbone(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(13, 4.2), sharey=True)
    top = title(fig, "Validation loss bottoms out around epochs 4-7 for every model, then rises",
                "Line = mean over seeds, band = range across seeds (SigLIP fine-tuned: 1 seed, no band). Dot: lowest mean "
                "validation loss. Each run keeps its best-AUROC epoch, so the later overfitting is not in the saved models.",
                below_pt=30)
    for ax, (name, frozen, tuned) in zip(axes, BACKBONES):
        ax.set_title(name, pad=10, fontsize=11)
        for exp, color, label in ((frozen, BLUE_LIGHT, "Frozen encoder"), (tuned, BLUE_DARK, "Last 4 blocks fine-tuned")):
            runs = sorted(root.glob(f"{exp}_10_epochs_s*/metrics.json"))
            losses = np.array([[e["val_loss"]["total"] for e in json.loads(r.read_text())["history"]] for r in runs])
            epochs = np.arange(1, losses.shape[1] + 1)
            if len(runs) > 1:
                ax.fill_between(epochs, losses.min(0), losses.max(0), color=color, alpha=0.18, linewidth=0)
            mean = losses.mean(0)
            ax.plot(epochs, mean, color=color, linewidth=2, label=label, solid_capstyle="round")
            best = int(mean.argmin())
            ax.scatter([epochs[best]], [mean[best]], s=60, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Epoch")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Validation loss")
    axes[-1].legend(loc="upper right", fontsize=9.5)
    fig.subplots_adjust(top=top, wspace=0.08, left=0.06, right=0.99, bottom=0.14)
    fig.savefig(out / "3_validation_loss_by_backbone.png", dpi=200)
    plt.close(fig)


def fig_confusion(root: Path, data_root: Path, out: Path, exp: str, label: str, filename: str) -> None:
    records = {r["sample_id"]: r for r in json.loads((data_root / "master_dataset_with_benign.json").read_text())}
    names = [op for _, ops in FAMILIES for op in ops]
    index = {op: i for i, op in enumerate(names)}
    counts = np.zeros((len(names), len(names)))
    runs = sorted(root.glob(f"{exp}_10_epochs_s*/best_auroc/val_predictions.jsonl"))
    for run in runs:
        for line in open(run):
            p = json.loads(line)
            r = records[p["sample_id"]]
            if r["label"] == "regression":
                counts[index[r["operator"]], index[p["op_pred"]]] += 1
    share = counts / counts.sum(1, keepdims=True)
    family_of = {op: fam for fam, ops in FAMILIES for op in ops}
    same_family = sum(counts[i, j] for i in range(len(names)) for j in range(len(names))
                      if family_of[names[i]] == family_of[names[j]]) / counts.sum()
    exact = np.trace(counts) / counts.sum()

    fig, ax = plt.subplots(figsize=(10.5, 9))
    off = [(share[i, j], names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j]
    worst, true_name, predicted_name = max(off)
    seeds = f"{len(runs)} seeds pooled" if len(runs) > 1 else "1 seed"
    top = title(fig, f"{label}: most type errors are between similar-looking changes",
                f"Mutation type predicted for each true type (rows sum to 100%), {label}, {seeds} "
                f"({int(counts.sum()):,} regression pairs). Exact type correct: {exact:.0%}; correct family: "
                f"{same_family:.0%}. Most common mistake: {true_name.replace('_', ' ')} predicted as "
                f"{predicted_name.replace('_', ' ')} ({worst:.0%}). Boxes outline the five families.", below_pt=10)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f6f9fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    image = ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if share[i, j] >= 0.1:  # label only the cells worth reading
                ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if share[i, j] >= 0.45 else INK)
    start = 0
    for _, ops in FAMILIES:
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), len(ops), len(ops), fill=False,
                               edgecolor=INK_2, linewidth=1.2))
        start += len(ops)
    labels = [n.replace("_", " ") for n in names]
    ax.set_xticks(range(len(names)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), labels)
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("True type")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    fig.subplots_adjust(top=top, left=0.17, right=0.92, bottom=0.17)
    fig.savefig(out / filename, dpi=200)
    plt.close(fig)


def fig_runtime(root: Path, out: Path) -> None:
    starts, minutes = {}, defaultdict(list)
    pattern = re.compile(r"^(\S+ \S+) (start|done)\s+(\S+)$")
    for log in ("all.log", "clip_siglip.log"):
        path = root / "logs" / log
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            stamp, event, run = match.groups()
            t = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            if event == "start":
                starts[run] = t
            elif run in starts and run not in CACHE_BUILDS:
                minutes[re.sub(r"_10_epochs_s\d+$", "", run)].append((t - starts[run]).total_seconds() / 60)
    rows = [("UIClip, frozen", "uiclip_cal", True), ("CLIP, frozen", "clip_cal", True),
            ("SigLIP-512, frozen", "siglip_cal", True), ("UIClip, fine-tuned", "uiclip_cal_ft4", True),
            ("CLIP, fine-tuned", "clip_cal_ft4", True), ("SigLIP-512, fine-tuned", "siglip_cal_ft4", False)]
    rows = [r for r in rows if minutes.get(r[1])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    top = title(fig, "SigLIP takes about 4x longer than UIClip/CLIP, even with the cache",
                "Wall-clock minutes per 10-epoch run on the RTX 5070 Ti (mean over seeds; runs that also built a feature cache "
                "are excluded). SigLIP fine-tuning ran without the cache.", below_pt=12)
    longest = max(statistics.mean(minutes[e]) for _, e, _ in rows)
    ax.set_xlim(0, longest * 1.15)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for i, (label, exp, cached) in enumerate(rows):
        y = len(rows) - 1 - i
        value = statistics.mean(minutes[exp])
        hbar(ax, y, value, 0.5, BLUE if cached else ORANGE)
        text = f"{value:.0f} min" if value < 120 else f"{value / 60:.1f} h"
        ax.text(value + longest * 0.012, y, text, va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows) - 1, -1, -1), [r[0] for r in rows])
    ax.set_xlabel("Minutes per 10-epoch run")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [matplotlib.patches.Patch(color=BLUE, label="With feature cache"),
               matplotlib.patches.Patch(color=ORANGE, label="Without feature cache")]
    ax.legend(handles=handles, loc="upper right", fontsize=9.5)
    fig.subplots_adjust(top=top, left=0.2, right=0.97, bottom=0.14)
    fig.savefig(out / "5_training_time.png", dpi=200)
    plt.close(fig)


GROUPS = ["first_batch", "all_experiments"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--data-root", default="d2c_v2")
    parser.add_argument("--out", default="docs/figures/uiclip", help="parent of the first_batch/ and all_experiments/ folders")
    parser.add_argument("--identical-seed", type=int, default=0, help="seed of the runs in the identical-pairs figure")
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=GROUPS, help="which figure folder(s) to draw")
    args = parser.parse_args()

    root, data_root = Path(args.outputs), Path(args.data_root)
    style()
    results = load_results(root)
    needed = set(EXPERIMENTS) if "first_batch" in args.only else set()
    if "all_experiments" in args.only:
        needed |= set(ALL_EXPERIMENTS)
    missing = sorted(e for e in needed if not results.get(e))
    if missing:
        raise SystemExit(f"no scored runs in {root}/comparison for: {missing}")

    if "first_batch" in args.only:
        out = Path(args.out) / "first_batch"
        out.mkdir(parents=True, exist_ok=True)
        fig_experiments(out)
        fig_benign(results, out)
        fig_metrics(results, out)
        fig_identical(root, data_root, out, args.identical_seed)
        fig_types(results, out)
        fig_curves(root, out)
    if "all_experiments" in args.only:
        out = Path(args.out) / "all_experiments"
        out.mkdir(parents=True, exist_ok=True)
        fig_all_experiments(out)
        fig_backbones(results, out)
        fig_curves_by_backbone(root, out)
        for suffix, (exp, label) in zip("abc", [("uiclip_cal_ft4", "UIClip fine-tuned"), ("clip_cal_ft4", "CLIP fine-tuned"),
                                                ("siglip_cal_ft4", "SigLIP-512 fine-tuned")]):
            fig_confusion(root, data_root, out, exp, label, f"4{suffix}_type_confusion_{label.split()[0].split('-')[0].lower()}.png")
        fig_runtime(root, out)
    print(f"drew {', '.join(args.only)} figures under {args.out}")


if __name__ == "__main__":
    main()
