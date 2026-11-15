"""Draw how every team model goes through the same test, as a slide-ready PNG.

Writes docs/figures/model_comparison_overview.png: 16:9 with a transparent background.

  python scripts/plot_model_comparison_overview.py

Requires matplotlib (``pip install -e ".[figures]"``) and the dataset in d2c_v2/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from plot_uiclip_findings import AQUA, AXIS, BLUE, INK, INK_2, ORANGE, style  # noqa: E402

OUT = Path("docs/figures/model_comparison_overview.png")
SCREENSHOTS = Path("d2c_v2/screenshots")
REFERENCE, CANDIDATE = "d2c_430.png", "d2c_430_background_color.png"
CROP = (0, 0, 1280, 800)

BLUE_TINT, AQUA_TINT, ORANGE_TINT, GREY_TINT = "#e3eefb", "#ddf3ea", "#fde6dc", "#efeee9"
FAMILIES = [  # heading, models, fill, edge
    ("UI-aware", "UIClip", BLUE_TINT, BLUE),
    ("Foundation", "CLIP, SigLIP, DINOv2, DINOv3", AQUA_TINT, AQUA),
    ("Change detection", "ChangeFormer, BIT, FC-Siam-Diff", ORANGE_TINT, ORANGE),
]


def rounded(ax, x, y, w, h, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=fill, edgecolor=edge, linewidth=2))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=20, color=INK_2,
                                 linewidth=2, shrinkA=0, shrinkB=0))


def screenshot(ax, name, x, y, w, h, label):
    image = Image.open(SCREENSHOTS / name).convert("RGB").crop(CROP)
    ax.imshow(image, extent=(x, x + w, y, y + h), interpolation="lanczos", zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=AXIS, linewidth=1.2, zorder=3))
    ax.text(x + w / 2, y + h + 1.4, label, ha="center", va="bottom", fontsize=15, color=INK)


def main() -> None:
    style()
    fig = plt.figure(figsize=(13.33, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])

    # The screenshot pair.
    pair_x, pair_w, pair_h = 2, 25, 15.6
    screenshot(ax, REFERENCE, pair_x, 41, pair_w, pair_h, "Reference")
    screenshot(ax, CANDIDATE, pair_x, 16.5, pair_w, pair_h, "Candidate")
    pair_right = pair_x + pair_w + 1.5

    # The three model families.
    fam_x, fam_w, fam_h = 37, 45, 17
    for i, (heading, models, fill, edge) in enumerate(FAMILIES):
        y = 51 - i * 22
        rounded(ax, fam_x, y, fam_w, fam_h, fill, edge)
        ax.text(fam_x + fam_w / 2, y + 11.2, heading, ha="center", va="center", fontsize=20,
                fontweight="bold", color=INK)
        ax.text(fam_x + fam_w / 2, y + 5.3, models, ha="center", va="center", fontsize=15, color=INK_2)
        arrow(ax, (pair_right, 37.5), (fam_x - 0.5, y + fam_h / 2))
        arrow(ax, (fam_x + fam_w + 0.5, y + fam_h / 2), (89.5, 37.5))

    # The shared scorer, then the comparison.
    rounded(ax, 90, 29.5, 16, 16, GREY_TINT, AXIS)
    ax.text(98, 37.5, "Same\nscorer", ha="center", va="center", fontsize=20, fontweight="bold",
            color=INK, linespacing=1.25)
    arrow(ax, (106.5, 37.5), (110.5, 37.5))
    rounded(ax, 111, 29.5, 20.5, 16, GREY_TINT, AXIS)
    ax.text(121.25, 37.5, "Side-by-side\nresults", ha="center", va="center", fontsize=17,
            fontweight="bold", color=INK, linespacing=1.25)

    ax.set_xlim(0, 133.3)
    ax.set_ylim(0, 75)
    ax.set_aspect("equal")
    ax.axis("off")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, transparent=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
