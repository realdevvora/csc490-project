"""Write the best checkpoint's predictions for every pair, in the team format (docs/prediction-format.md), to one folder.

    python3 save_predictions.py                            # newest run's ckpt_best.pt
    python3 save_predictions.py runs/<id>/ckpt_best.pt     # a specific checkpoint
    python3 save_predictions.py runs/<id>/ckpt_best.pt out # a different output folder

Run from models/fc_siam_diff. Writes fc_siam_diff_all.jsonl (every pair), fc_siam_diff_train.jsonl and
fc_siam_diff_val.jsonl to ../../outputs/fc_siam_diff/ by default. Each file covers exactly its subset, so it can be
passed straight to scripts/score_predictions.py with --subset all, train or val.
"""

import json
import sys
from pathlib import Path

import torch

# Run the notebook's own setup and prediction code, skipping the training cells, so predictions match training exactly
cells = [''.join(c['source']) for c in json.loads(Path('train_fc_siam_diff.ipynb').read_text())['cells'] if c['cell_type'] == 'code']
g = {}
for src in cells:
    if src.startswith(('def save', 'for epoch')):
        continue
    if src.startswith('def write_predictions'):
        src = src.split('\nnet.load_state_dict')[0]
    exec(src, g)

ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else max(Path('runs').glob('*/ckpt_best.pt'), key=lambda p: p.parent.name)
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('../../outputs/fc_siam_diff')
out.mkdir(parents=True, exist_ok=True)
state = torch.load(ckpt, map_location=g['device'])
g['net'].load_state_dict(state['model'])
g['run_dir'] = ckpt.parent
print('checkpoint', ckpt, 'pass', state['epoch'] + 1, flush=True)

# Predict every pair once, then split the lines into the train and val subsets
all_path = out / 'fc_siam_diff_all.jsonl'
g['write_predictions'](g['rows'], all_path)
lines = {json.loads(line)['sample_id']: line for line in all_path.read_text().splitlines()}
assert sorted(lines) == sorted(r['sample_id'] for r in g['rows'])
for name, rows in (('train', g['train_rows']), ('val', g['val_rows'])):
    ids = [r['sample_id'] for r in rows] + (sorted(g['BLANK_RENDERS']) if name == 'train' else [])
    (out / f'fc_siam_diff_{name}.jsonl').write_text(''.join(lines[i] + '\n' for i in ids))
    print(name, len(ids), 'predictions')
print('all', len(lines), 'predictions; written to', out.resolve())
