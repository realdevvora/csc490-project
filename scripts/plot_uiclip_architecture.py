"""Draw the Siamese UIClip/CLIP/SigLIP detector architecture as a slide-ready PNG.

Mirrors ``regression_lens.models.uiclip.SiameseDetector`` and the tiling in
``regression_lens.data.pairs``. Writes two versions to docs/figures/uiclip/all_experiments/:

  0_model_architecture.png         annotated: title, captions, loss and backbone notes
  0b_model_architecture_simple.png diagram only: stage labels, inputs, block names

  python scripts/plot_uiclip_architecture.py

Requires matplotlib (``pip install -e ".[figures]"``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from plot_uiclip_findings import (  # noqa: E402
    AQUA, AXIS, BLUE, BLUE_DARK, GRID, INK, INK_2, MUTED, ORANGE, SURFACE, style, title,
)

OUT_DIR = Path("docs/figures/uiclip/all_experiments")
OUT, OUT_SIMPLE = "0_model_architecture.png", "0b_model_architecture_simple.png"

BLUE_TINT, AQUA_TINT, ORANGE_TINT, GREY_TINT = "#e3eefb", "#ddf3ea", "#fde6dc", "#efeee9"


def box(ax, x, y, w, h, heading, body="", fill=GREY_TINT, edge=AXIS, dashed=False, size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=1.4,
                                linestyle=(0, (4, 3)) if dashed else "-"))
    if body:
        ax.text(x + w / 2, y + h - 2.3, heading, ha="center", va="center", fontsize=size,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, (y + h - 4.2 + y) / 2, body, ha="center", va="center", fontsize=8.8,
                color=INK_2, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, heading, ha="center", va="center", fontsize=size,
                fontweight="bold", color=INK)


def arrow(ax, start, end, color=INK_2, dashed=False, rad=0.0, label=None, label_xy=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=color,
                                 linewidth=1.4, linestyle=(0, (4, 3)) if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=8.5, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))


def page(ax, x, y, w, h, changed=False):
    """A stylized screenshot cut into aligned tiles (6 columns x 4 rows)."""

    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=AXIS, linewidth=1.2))
    # page content: header bar, text lines, a card
    ax.add_patch(Rectangle((x + 0.06 * w, y + 0.82 * h), 0.88 * w, 0.1 * h, facecolor=BLUE_DARK, lw=0))
    for i in range(4):
        ax.add_patch(Rectangle((x + 0.08 * w, y + (0.68 - 0.08 * i) * h), (0.55 - 0.07 * i) * w, 0.03 * h,
                               facecolor=AXIS, lw=0))
    card = ORANGE if changed else BLUE
    ax.add_patch(Rectangle((x + (0.58 if changed else 0.55) * w, y + 0.12 * h), 0.36 * w, 0.28 * h,
                           facecolor=card, lw=0, alpha=0.85))
    for c in range(1, 6):
        ax.plot([x + c * w / 6] * 2, [y, y + h], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    for r in range(1, 4):
        ax.plot([x, x + w], [y + r * h / 4] * 2, color=MUTED, lw=0.7, ls=(0, (2, 2)))


def tokens(ax, x, y, size, color, n=5):
    """A small grid of patch tokens."""

    step = size / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + i * step, y + j * step), step * 0.82, step * 0.82,
                                   facecolor=color, alpha=0.35 + 0.5 * ((i * 3 + j * 7) % 5) / 4, lw=0))


def change_map(ax, x, y, w, h):
    """A page-level heatmap with the largest-component box and peak point."""

    rows, cols = 8, 12
    yy, xx = np.mgrid[0:rows, 0:cols]
    heat = np.exp(-(((xx - 8.3) / 1.6) ** 2 + ((yy - 2.0) / 1.2) ** 2))
    ax.imshow(heat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("m", ["white", ORANGE]),
              extent=(x, x + w, y, y + h), origin="upper", vmin=0, vmax=1, zorder=2,
              interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=AXIS, lw=1.2, zorder=3))
    cw, ch = w / cols, h / rows
    ax.add_patch(Rectangle((x + 6.5 * cw, y + (rows - 4) * ch), 4 * cw, 3 * ch, facecolor="none",
                           edgecolor=INK, lw=1.4, zorder=4))
    ax.plot(x + 8.8 * cw, y + (rows - 2.5) * ch, marker="+", color=INK, ms=9, mew=2, zorder=5)


def draw(out: Path, simple: bool = False) -> None:
    """Draw the architecture; ``simple`` keeps only the diagram, without captions or notes."""

    def body(text: str) -> str:
        return "" if simple else text

    def caption(*args, **kwargs) -> None:
        if not simple:
            ax.text(*args, **kwargs)

    style()
    if simple:
        y0 = 29
        fig = plt.figure(figsize=(17, 17 * (93 - y0) / 170))
        ax = fig.add_axes((0.005, 0.005, 0.99, 0.99))
        ax.set_ylim(y0, 93)
    else:
        fig = plt.figure(figsize=(17, 10))
    top = 1.0 if simple else title(
        fig,
        "Siamese change detector: one shared vision encoder, three heads",
        "Reference and candidate screenshots are cut into aligned tiles and embedded by the same "
        "encoder (UIClip, CLIP or SigLIP). Their patch tokens are fused and read by a patch head "
        "(where), an image head (whether) and a type head (which CSS change). Patch and image "
        "scores are self-calibrated against the reference compared with itself.",
    )
    if not simple:
        ax = fig.add_axes((0.01, 0.01, 0.98, top - 0.01))
        ax.set_ylim(0, 92)
    ax.set_xlim(0, 170)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- column labels ------------------------------------------------------
    for x, label in [(12, "1  Tile"), (40, "2  Encode"), (71, "3  Fuse"),
                     (106, "4  Heads"), (148, "5  Outputs")]:
        ax.text(x, 90.5, label, ha="center", va="center", fontsize=12, fontweight="bold", color=BLUE_DARK)
    ax.plot([2, 168], [88, 88], color=GRID, lw=1)

    # ---- 1 inputs -----------------------------------------------------------
    ax.text(12, 84.5, "Reference", ha="center", fontsize=10, fontweight="bold", color=INK)
    page(ax, 2, 66, 20, 17)
    ax.text(12, 60.5, "Candidate", ha="center", fontsize=10, fontweight="bold", color=INK)
    page(ax, 2, 42, 20, 17, changed=True)
    caption(12, 38.5, "1280 px wide, any height\nresized to 6 tile columns,\n"
            "padded white, cut into\nT aligned tiles\n(224 px; SigLIP 512 px)",
            ha="center", va="top", fontsize=8.8, color=INK_2, linespacing=1.3)

    # ---- 2 shared encoder ---------------------------------------------------
    ex, ew = 28, 24
    ax.add_patch(FancyBboxPatch((ex, 40), ew, 44, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=BLUE_TINT, edgecolor=BLUE, linewidth=1.6))
    ax.text(ex + ew / 2, 81.3, "Shared ViT encoder", ha="center", fontsize=11, fontweight="bold", color=INK)
    caption(ex + ew / 2, 78.3, "same weights for both images", ha="center", fontsize=8.8, color=INK_2)
    box(ax, ex + 2, 69, ew - 4, 6.5, "Patch embedding", body("+ position, pre-norm"), fill="white", size=9.5)
    box(ax, ex + 2, 57, ew - 4, 10, "Blocks 1 – 8", body("always frozen\nfeatures cached on disk"),
        fill=GREY_TINT, size=9.5)
    box(ax, ex + 2, 43, ew - 4, 12, "Blocks 9 – 12", body("frozen, or fine-tuned\nin the \"last 4 blocks\" runs\n"
        "(lr 1e-5)"), fill=ORANGE_TINT, edge=ORANGE, size=9.5)
    arrow(ax, (ex + ew / 2, 69), (ex + ew / 2, 67))
    arrow(ax, (ex + ew / 2, 57), (ex + ew / 2, 55))
    arrow(ax, (22, 74.5), (ex, 72.5), color=BLUE)
    arrow(ax, (22, 50.5), (ex, 70.5), color=BLUE, rad=0.0)

    # ---- patch tokens a, b --------------------------------------------------
    tokens(ax, 56, 66, 9, BLUE)
    tokens(ax, 56, 47, 9, ORANGE)
    ax.text(60.5, 77.5, "a  (reference)", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(60.5, 58.5, "b  (candidate)", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    caption(60.5, 44.5, "patch tokens\nT × P × 768\n(CLS dropped)", ha="center", va="top",
            fontsize=8.8, color=INK_2, linespacing=1.3)
    arrow(ax, (ex + ew, 62), (56, 70.5), color=BLUE)
    arrow(ax, (ex + ew, 60), (56, 51.5), color=BLUE)

    # ---- 3 fusion -----------------------------------------------------------
    fx, fw = 69, 24
    box(ax, fx, 63, fw, 13, "Pair fusion  fuse(a, b)",
        body("[ a,  b,  |a − b|,  a ⊙ b ]   (3072)\nMLP 3072 → 512 → GELU → 256\n→ fused tokens  T × P × 256"),
        fill=AQUA_TINT, edge=AQUA, size=10)
    box(ax, fx, 44, fw, 10, "Self-pair  fuse(a, a)",
        body("same MLP, reference vs itself\nreuses a; only the heads rerun"),
        fill=SURFACE, edge=AQUA, dashed=True, size=10)
    arrow(ax, (65, 70.5), (fx, 70.5), color=INK_2)
    arrow(ax, (65, 51.5), (fx, 66), color=INK_2)
    arrow(ax, (65, 70.5), (fx, 50), color=AQUA, dashed=True)

    # ---- 4 heads -------------------------------------------------------------
    hx, hw = 98, 20
    box(ax, hx, 74, hw, 8, "Patch head", body("Linear 256 → 1 per patch"), fill=ORANGE_TINT, edge=ORANGE)
    box(ax, hx, 58, hw, 9, "Attention pool", body("softmax(patch logits)\nweights the fused tokens"),
        fill=AQUA_TINT, edge=AQUA)
    box(ax, hx, 45, hw, 9, "Image head", body("Linear 512 → 1 on\n[attn-pooled, max-pooled]"),
        fill=ORANGE_TINT, edge=ORANGE)
    box(ax, hx, 32, hw, 8, "Type head", body("Linear 256 → 13 operators"), fill=ORANGE_TINT, edge=ORANGE)
    arrow(ax, (fx + fw, 72), (hx, 78))
    arrow(ax, (fx + fw, 68), (hx, 62.5))
    arrow(ax, (hx + hw / 2, 74), (hx + hw / 2, 67), label=None if simple else "weights", label_xy=(hx + hw / 2 + 4.5, 70.5))
    arrow(ax, (hx + hw / 2, 58), (hx + hw / 2, 54))
    # attention-pooled vector also feeds the type head (routed left of the heads)
    lx = hx - 2.5
    ax.plot([hx, lx, lx], [60, 60, 36], color=INK_2, lw=1.4)
    arrow(ax, (lx, 36), (hx, 36))
    caption(lx - 1, 39, "pooled", rotation=90, ha="center", va="bottom", fontsize=8, color=INK_2)

    # self-calibration nodes
    cx = hx + hw + 5
    for y in (78, 49.5):
        ax.add_patch(plt.Circle((cx, y), 2.6, facecolor="white", edgecolor=AQUA, lw=1.6, zorder=3))
        ax.text(cx, y, "−", ha="center", va="center", fontsize=14, fontweight="bold", color=AQUA, zorder=4)
        arrow(ax, (hx + hw, y), (cx - 2.6, y))
    # self-pair path: through the gap between image and type heads, then up a bus
    dash = (0, (4, 3))
    bus_y, bus_x = 42.5, cx + 4.5
    ax.plot([fx + fw, fx + fw + 2, fx + fw + 2, bus_x, bus_x], [47, 47, bus_y, bus_y, 72],
            color=AQUA, lw=1.4, ls=dash)
    arrow(ax, (bus_x, 72), (cx + 1.6, 75.9), color=AQUA, dashed=True)
    arrow(ax, (cx, bus_y), (cx, 49.5 - 2.6), color=AQUA, dashed=True)
    caption((hx + hw + bus_x) / 2 - 6, bus_y + 1.1, "self-pair through the same heads",
            ha="center", va="bottom", fontsize=8, color=AQUA)
    caption(cx, 84.5, "self-calibration\nhead(a, b) − head(a, a) + offset",
            ha="center", va="center", fontsize=8.6, color=AQUA, fontweight="bold", linespacing=1.3)

    # ---- 5 outputs -----------------------------------------------------------
    ox = 132
    change_map(ax, ox, 70, 16, 11)
    ax.text(ox + 18, 78.5, "Change map", ha="left", fontsize=10, fontweight="bold", color=INK)
    caption(ox + 18, 76.5, "sigmoid per patch,\nregrouped into the page\n→ box (largest region\n"
            "above 0.5) and peak\npoint  (where)", ha="left", va="top", fontsize=8.6, color=INK_2,
            linespacing=1.25)
    arrow(ax, (hx + hw + 7.6, 78), (ox, 76))

    box(ax, ox, 45, 16, 9, "p(regression)", body("sigmoid, one per pair"), fill=BLUE_TINT, edge=BLUE, size=10)
    caption(ox + 18, 49.5, "whether the\ncandidate regressed", ha="left", va="center", fontsize=8.6,
            color=INK_2, linespacing=1.25)
    arrow(ax, (hx + hw + 7.6, 49.5), (ox, 49.5))

    box(ax, ox, 32, 16, 8, "Mutation type", body("13-way softmax"), fill=BLUE_TINT, edge=BLUE, size=10)
    caption(ox + 18, 36, "which CSS change:\ncolour, border, font,\nmargin, padding, …", ha="left",
            va="center", fontsize=8.6, color=INK_2, linespacing=1.25)
    arrow(ax, (hx + hw, 36), (ox, 36))

    if not simple:
        notes(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def notes(ax) -> None:
    """Bottom band of the annotated version: loss, why self-calibrate, backbones."""

    ax.plot([2, 168], [27, 27], color=GRID, lw=1)
    ax.text(2, 24, "Training loss", ha="left", fontsize=11, fontweight="bold", color=INK)
    ax.text(2, 20.5,
            "BCE(patch logits, patches inside diff_bbox; pos weight 5)\n"
            "+ BCE(image logit, is regression)\n"
            "+ 0.5 × CE(type logits, operator)   regressions only\n\n"
            "AdamW: heads lr 1e-3, encoder lr 1e-5; 8 pages per step;\n"
            "trained on regression, no-change and benign pairs",
            ha="left", va="top", fontsize=9, color=INK_2, linespacing=1.35)

    ax.text(62, 24, "Why self-calibrate?", ha="left", fontsize=11, fontweight="bold", color=INK)
    ax.text(62, 20.5,
            "Two identical screenshots give head(a, a) − head(a, a) = 0,\n"
            "so every identical pair scores exactly the learned offset,\n"
            "on every page. The heads still see page content, and the\n"
            "self-pass reuses the encoder output, so it is nearly free.",
            ha="left", va="top", fontsize=9, color=INK_2, linespacing=1.35)

    ax.text(113, 24, "Backbones tried", ha="left", fontsize=11, fontweight="bold", color=INK)
    rows = [("", "tile", "patches / tile", "CLS"),
            ("UIClip  ViT-B/32", "224", "7 × 7 = 49", "yes"),
            ("CLIP  ViT-B/32", "224", "7 × 7 = 49", "yes"),
            ("SigLIP  B/16-512", "512", "32 × 32 = 1024", "no")]
    xs = [113, 135, 145, 162]
    for i, row in enumerate(rows):
        y = 20 - i * 3.6
        for j, (x, cell) in enumerate(zip(xs, row)):
            ax.text(x, y, cell, ha="left" if j == 0 else "center", va="center", fontsize=9,
                    color=MUTED if i == 0 else (INK if j == 0 else INK_2),
                    fontweight="bold" if j == 0 and i else "normal")
    ax.text(113, 4.5, "All three: 12 blocks, width 768; each run either frozen or last 4 blocks trained.",
            ha="left", va="center", fontsize=8.6, color=INK_2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    out_dir = parser.parse_args().out_dir
    draw(out_dir / OUT)
    draw(out_dir / OUT_SIMPLE, simple=True)


if __name__ == "__main__":
    main()
