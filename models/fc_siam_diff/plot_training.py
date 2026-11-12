"""Plot how a training run's validation metrics changed pass by pass.

    python3 plot_training.py                          # newest run in runs/
    python3 plot_training.py runs/<id>/metrics.jsonl  # a specific run

Writes PNGs to runs/<id>/plots/. The dashed line marks the pass kept as the best checkpoint (highest pixel F1).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Reference palette: three categorical hues validated for colour-blind separation, plus neutral inks
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK_2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
    'axes.titlesize': 12, 'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
    'font.size': 10, 'legend.frameon': False, 'legend.labelcolor': INK_2, 'lines.linewidth': 2,
})


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    best = max(rows, key=lambda r: r['pixel_f1'])['epoch']
    return rows, best


def mark_best(ax, best):
    ax.axvline(best, color=MUTED, linewidth=1, linestyle='--', zorder=1)


# One chart with a few series on a shared 0-100% axis, each labelled at its last point as well as in the legend
def line_chart(rows, best, series, title, path, percent=True):
    x = [r['epoch'] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for label, key, colour in series:
        y = [r[key] for r in rows]
        ax.plot(x, y, color=colour, label=label, solid_capstyle='round')
        ax.annotate(f'{y[-1]:.0%}' if percent else f'{y[-1]:.2f}', (x[-1], y[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=9, color=INK)
    if percent:
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel('Training pass')
    ax.set_title(title, pad=20)
    ax.text(0, 1.03, f'Dashed line: best checkpoint (pass {best})', transform=ax.transAxes, fontsize=8.5, color=MUTED)
    mark_best(ax, best)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=len(series))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# One small panel per regression type, sorted by final accuracy, all on the same 0-100% scale
def per_type_chart(rows, best, path):
    x = [r['epoch'] for r in rows]
    ops = sorted(rows[-1]['type_accuracy_by_op'], key=lambda o: -rows[-1]['type_accuracy_by_op'][o])
    cols = 5
    fig, axes = plt.subplots((len(ops) + cols - 1) // cols, cols, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, op in zip(axes.flat, ops):
        y = [r['type_accuracy_by_op'].get(op, 0) for r in rows]
        ax.plot(x, y, color=BLUE, linewidth=1.6)
        ax.set_title(f'{op}  {y[-1]:.0%}', fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        mark_best(ax, best)
    for ax in list(axes.flat)[len(ops):]:
        ax.axis('off')
    fig.suptitle(f'Right type named, by regression type (dashed line: best checkpoint, pass {best})', x=0.01, ha='left',
                 fontweight='bold', color=INK)
    fig.supxlabel('Training pass', color=INK_2, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(Path('runs').glob('*/metrics.jsonl'), key=lambda p: p.parent.name)
    rows, best = load(path)
    out = path.parent / 'plots'
    out.mkdir(exist_ok=True)
    line_chart(rows, best, [('Regression pages flagged', 'image_recall', BLUE),
                            ('Benign pages flagged', 'image_fpr_benign', ORANGE),
                            ('No-change pages flagged', 'image_fpr_no_change', AQUA)],
               'Pages with any flagged pixel', out / 'detection.png')
    line_chart(rows, best, [('Type accuracy (13 types)', 'type_accuracy', BLUE),
                            ('Family accuracy (5 families)', 'family_accuracy', ORANGE)],
               'Right regression type named', out / 'type_accuracy.png')
    line_chart(rows, best, [('Pixel precision', 'pixel_precision', BLUE),
                            ('Pixel recall', 'pixel_recall', ORANGE),
                            ('Pixel F1', 'pixel_f1', AQUA)],
               'Edited pixels marked with a regression type', out / 'pixels.png')
    line_chart(rows, best, [('Training loss', 'train_loss', BLUE), ('Validation loss', 'val_loss', ORANGE)],
               'Loss', out / 'loss.png', percent=False)
    per_type_chart(rows, best, out / 'type_accuracy_by_type.png')
    print(f'{len(rows)} passes, best checkpoint at pass {best}; wrote', ', '.join(sorted(p.name for p in out.glob('*.png'))), 'to', out)


if __name__ == '__main__':
    main()
