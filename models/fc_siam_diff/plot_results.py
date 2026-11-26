"""Chart a prediction file's final results against the dataset labels.

    python3 plot_results.py                                    # results/fc_siam_diff_val.jsonl
    python3 plot_results.py <predictions.jsonl> [data_root]    # data_root holds master_dataset_with_benign.json

Writes PNGs to a plots/ folder next to the prediction file:
  scores.png             how page scores spread for regression, benign and no-change pages
  roc.png                regression vs benign: share caught against share of benign pages falsely flagged
  detection_by_type.png  per type, how well its pages are ranked above benign pages (AUROC)
  type_f1.png            per type, how well the type is named (F1)
  confusion.png          per type, which type the model names instead
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

# Reference palette: three categorical hues, a light-to-dark blue ramp for amounts, and neutral inks
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
BLUE_RAMP = LinearSegmentedColormap.from_list('blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
SURFACE, INK, INK_2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
LABELS = (('regression', 'Regression', BLUE), ('benign', 'Benign (browser difference)', ORANGE), ('no_change', 'No change', AQUA))

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
    'axes.titlesize': 12, 'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
    'font.size': 10, 'legend.frameon': False, 'legend.labelcolor': INK_2, 'lines.linewidth': 2,
})


# Join each prediction with its pair's label and true type from the dataset metadata
def load(pred_path, data_root):
    meta = {r['sample_id']: r for r in json.loads((data_root / 'master_dataset_with_benign.json').read_text())}
    preds = [json.loads(line) for line in pred_path.read_text().splitlines() if line.strip()]
    return [{**p, 'label': meta[p['sample_id']]['label'], 'op': meta[p['sample_id']]['operator']} for p in preds]


# Share of (positive, negative) pairs where the positive scores higher; ties count half
def auroc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def f1(rows, op):
    tp = sum(r['op'] == op and r['op_pred'] == op for r in rows)
    called, actual = sum(r['op_pred'] == op for r in rows), sum(r['op'] == op for r in rows)
    return 2 * tp / (called + actual) if called + actual else 0.0


def finish(fig, ax, title, subtitle, path):
    ax.set_title(title, pad=20)
    ax.annotate(subtitle, (0, 1), xytext=(0, 6), xycoords='axes fraction', textcoords='offset points', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# One histogram per label on a shared score axis, with the score that catches 95% of regressions marked
def scores_chart(rows, path):
    cut = np.percentile([r['score'] for r in rows if r['label'] == 'regression'], 5)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)
    bins = np.linspace(0, 1, 41)
    for ax, (label, name, colour) in zip(axes, LABELS):
        s = [r['score'] for r in rows if r['label'] == label]
        ax.hist(s, bins=bins, color=colour, weights=np.full(len(s), 1 / len(s)), edgecolor=SURFACE, linewidth=1)
        ax.axvline(cut, color=MUTED, linewidth=1, linestyle='--')
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=None))
        ax.set_title(f'{name}  (n={len(s)}, median {np.median(s):.2f})', fontsize=10, fontweight='normal', pad=4)
        ax.grid(axis='x', visible=False)
    axes[-1].set_xlabel('Page score (mean regression confidence over the pixels that differ)')
    fig.supylabel('Share of pages', color=INK_2, fontsize=10)
    fig.suptitle('How page scores spread by label', x=0.02, ha='left', fontweight='bold', color=INK)
    fig.text(0.02, 0.925, f'Dashed line: score that catches 95% of regressions ({cut:.2f})', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def roc_chart(rows, path):
    pos = np.array([r['score'] for r in rows if r['label'] == 'regression'])
    neg = np.array([r['score'] for r in rows if r['label'] == 'benign'])
    cuts = np.r_[np.inf, np.unique(np.r_[pos, neg])[::-1]]
    tpr, fpr = [(pos >= c).mean() for c in cuts], [(neg >= c).mean() for c in cuts]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle='--')
    ax.plot(fpr, tpr, color=BLUE)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.01)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel('Benign pages flagged')
    ax.set_ylabel('Regressions caught')
    finish(fig, ax, 'Regression vs benign', f'AUROC {auroc(pos, neg):.3f}; dashed line: random guessing (0.5)', path)


# Horizontal bars sorted best to worst, value written at the end of each bar, with an optional reference line
def type_bars(values, counts, title, subtitle, xlabel, path, ref=None):
    ops = sorted(values, key=values.get)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(ops, [values[o] for o in ops], color=BLUE, height=0.7)
    for i, o in enumerate(ops):
        ax.text(values[o] + 0.01, i, f'{values[o]:.2f}', va='center', fontsize=9, color=INK)
    ax.set_yticks(range(len(ops)), [f'{o} ({counts[o]})' for o in ops])
    if ref is not None:
        ax.axvline(ref, color=MUTED, linewidth=1, linestyle='--')
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    finish(fig, ax, title, subtitle, path)


# Rows: true type; columns: type named. Each row sums to 100%
def confusion_chart(reg, ops, path):
    m = np.array([[sum(r['op'] == t and r['op_pred'] == p for r in reg) for p in ops] for t in ops], dtype=float)
    m /= m.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(m, cmap=BLUE_RAMP, vmin=0, vmax=1)
    for t in range(len(ops)):
        for p in range(len(ops)):
            if m[t, p] >= 0.005:
                ax.text(p, t, f'{m[t, p]:.0%}', ha='center', va='center', fontsize=7.5, color=SURFACE if m[t, p] > 0.5 else INK)
    ax.set_xticks(range(len(ops)), ops, rotation=45, ha='right')
    ax.set_yticks(range(len(ops)), ops)
    ax.set_xlabel('Type named by the model')
    ax.set_ylabel('True type')
    ax.grid(False)
    finish(fig, ax, 'What each regression type is named as', 'Each row is one true type and sums to 100%; the diagonal is correct', path)


def main():
    here = Path(__file__).parent
    pred_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / 'results' / 'fc_siam_diff_val.jsonl'
    data_root = Path(sys.argv[2]) if len(sys.argv) > 2 else here / '../../data/processed/d2c_v3'
    rows = load(pred_path, data_root)
    out = pred_path.parent / 'plots'
    out.mkdir(exist_ok=True)

    reg = [r for r in rows if r['label'] == 'regression']
    benign = [r['score'] for r in rows if r['label'] == 'benign']
    counts = Counter(r['op'] for r in reg)
    ops = sorted(counts)

    scores_chart(rows, out / 'scores.png')
    roc_chart(rows, out / 'roc.png')
    type_bars({o: auroc([r['score'] for r in reg if r['op'] == o], benign) for o in ops}, counts,
              'How well each type is told apart from benign pages',
              'AUROC of that type\'s pages vs benign pages; dashed line: random guessing (0.5)', 'AUROC',
              out / 'detection_by_type.png', ref=0.5)
    type_bars({o: f1(reg, o) for o in ops}, counts, 'How well each type is named',
              f'F1 per type on true regressions; macro-F1 {np.mean([f1(reg, o) for o in ops]):.3f}', 'F1',
              out / 'type_f1.png')
    confusion_chart(reg, ops, out / 'confusion.png')
    print(len(rows), 'predictions; wrote', ', '.join(sorted(p.name for p in out.glob('*.png'))), 'to', out)


if __name__ == '__main__':
    main()
