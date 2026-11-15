"""Plot learning diagnostics and predeclared seed-490 validation review cases."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from regression_lens.data.records import load_records
from regression_lens.models.dino_runtime import read, open_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    a = p.parse_args()
    selection = read(a.evaluation/'selection.json')
    config = read(a.evaluation/'config.json')
    review = a.evaluation/'review'
    review.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = dict(B='tab:blue', C='tab:orange')
    for condition in ('B', 'C'):
        for seed, style in zip((490, 491, 492), ('-', '--', ':')):
            rows = [json.loads(line) for line in (a.runs/condition/f'seed{seed}'/'learning.jsonl').read_text().splitlines()]
            for ax, key in zip(axes.flat[:3], ('loss', 'fixed_training_loss', 'gradient_norm')):
                ax.plot([r['step'] for r in rows], [r[key] for r in rows], style,
                        color=colors[condition], label=f'{condition}/{seed}')
                ax.set(xlabel='Update', ylabel=key)
        points = sorted((r for r in selection['checkpoint_candidates'] if r['condition'] == condition), key=lambda r: r['step'])
        axes[1, 1].errorbar([r['step'] for r in points], [r['auroc_mean'] for r in points],
            yerr=[r['auroc_std'] for r in points], color=colors[condition], label=condition, capsize=4)
    axes[1, 1].set(xlabel='Checkpoint update', ylabel='Full-page validation AUROC (mean ± sample SD)')
    for ax in axes.flat:
        ax.legend(fontsize=7)
    figure.savefig(review/'learning.png', dpi=140)
    plt.close(figure)
    records = {r['sample_id']: r for r in load_records(a.data_root)}
    indices = {sid: i for i, sid in enumerate(config['sample_ids'])}
    cases = []
    for condition in ('B', 'C'):
        chosen = selection['selected'][condition]['seeds']['490']
        root = a.evaluation/chosen['name']
        predictions = [json.loads(line) for line in (root/'predictions.jsonl').read_text().splitlines()]
        threshold = chosen['metrics']['threshold']
        benign = sorted((r for r in predictions if records[r['sample_id']]['label'] == 'benign'), key=lambda r: -r['score'])[:2]
        regression = sorted((r for r in predictions if records[r['sample_id']]['label'] == 'regression'), key=lambda r: r['score'])[:2]
        for index, prediction in enumerate(benign+regression):
            record = records[prediction['sample_id']]
            source = root/f"{indices[record['sample_id']]:05d}.npz"
            with np.load(source) as maps:
                values = np.ma.masked_where(~maps['valid'], maps['location'])
            images = open_pair(a.data_root, record)
            figure, axes = plt.subplots(1, 3, figsize=(12, 10), layout='constrained')
            for ax, image, title in zip(axes[:2], images, ('Reference', 'Candidate')):
                ax.imshow(image)
                ax.set_title(title)
            heat = axes[2].imshow(values, extent=(0, values.shape[1]*2, values.shape[0]*2, 0), vmin=0, vmax=1, cmap='magma')
            axes[2].set_title('Localization probability')
            if record['label'] == 'regression':
                box = record['bbox_after'] if record['reflow'] else record['diff_bbox']
                for ax in axes[1:]:
                    ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, edgecolor='lime'))
            axes[2].scatter(*prediction['point'], c='cyan', marker='+')
            for ax in axes:
                ax.set(xlim=(0, max(im.width for im in images)), ylim=(max(im.height for im in images), 0))
                ax.axis('off')
            flagged = prediction['score'] >= threshold
            error = flagged != (record['label'] == 'regression')
            figure.suptitle(f"{condition}/seed490 {record['sample_id']} | {record['label']} | score={prediction['score']:.4f} threshold={threshold:.4f} | {'error' if error else 'correct'}")
            figure.colorbar(heat, ax=axes[2], shrink=.5)
            filename = f'{condition}_{index}.png'
            figure.savefig(review/filename, dpi=120)
            plt.close(figure)
            cases.append(dict(condition=condition, seed=490, sample_id=record['sample_id'],
                score=prediction['score'], threshold=threshold, error=bool(error), figure=filename))
    (review/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')


if __name__ == '__main__':
    main()
