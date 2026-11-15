"""Combine completed pretrained and three-seed learned-head validation results."""
import argparse
from pathlib import Path
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baselines', type=Path, required=True)
    p.add_argument('--v2-heads', type=Path, required=True)
    p.add_argument('--v3-heads', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Choose an empty output directory')
    rows = []
    for name in ('dinov2', 'dinov3', 'pixel'):
        m = read(a.baselines/f'{name}.metrics.json')
        if m['n'] != 1033:
            raise ValueError('Expected complete committed validation set (1033 pairs)')
        rows.append(dict(model=name, condition='A' if name != 'pixel' else 'pixel',
                         metrics=m, auroc=m['auroc']['regression_vs_benign'], sd=None,
                         benign_fpr95=m['benign_fpr_at_95_recall']['fpr'],
                         pointing=m.get('localization', {}).get('pointing_accuracy')))
    for name, root in (('dinov2', a.v2_heads), ('dinov3', a.v3_heads)):
        selection = read(root/'selection.json')
        if selection['complete']['pairs'] != 1033:
            raise ValueError('Expected full validation head predictions')
        for condition in ('B', 'C'):
            row = selection['selected'][condition]
            rows.append(dict(model=name, condition=condition, step=row['step'],
                auroc=row['auroc_mean'], sd=row['auroc_std'],
                benign_fpr95=row['benign_fpr95_mean'],
                pointing=row['aggregate_metrics'].get('localization/pointing_accuracy', {}).get('mean'),
                metrics=row['aggregate_metrics'], seeds=row['seeds']))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'results.json').write_text(json.dumps(dict(results=rows,
        interpretation='Validation-selected checkpoints and thresholds; no independent test results'), indent=2)+'\n')
    lines = ['| Model | Condition | Reg/benign AUROC | Seed SD | Benign FPR@95R | Pointing |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        sd = '-' if row['sd'] is None else f"{row['sd']:.4f}"
        point = '-' if row['pointing'] is None else f"{row['pointing']:.4f}"
        lines.append(f"| {row['model']} | {row['condition']} | {row['auroc']:.4f} | {sd} | {row['benign_fpr95']:.4f} | {point} |")
    (a.out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
