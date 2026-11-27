"""Run a FC-Siam-diff checkpoint on every prepared DiffSpot pair (Test 1 base predictions).

Uses the training notebook's own prediction code, the same way
``export_fc_siam_predictions.py`` does. The notebook setup still reads the
Design2Code index for the type names, so ``--data-root`` must point at it.
Only ``inference.jsonl`` and its images are read from DiffSpot.

    python3 scripts/fc_siam_diffspot_predict.py runs/20261111-210302/ckpt_best.pt \\
        --dataset data/external/diffspot-full --out outputs/diffspot/fc-siam-base.jsonl \\
        [--data-root data/processed/d2c_v3]
"""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from export_fc_siam_predictions import NOTEBOOK, apply_config, export_cells

SHARED_FIELDS = ('sample_id', 'score', 'op_pred', 'box', 'point')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--dataset', type=Path, required=True, help='prepared DiffSpot folder')
    parser.add_argument('--out', type=Path, required=True, help='base prediction JSONL to write')
    parser.add_argument('--data-root', type=Path, help='Design2Code data folder, for the notebook setup')
    parser.add_argument('--num-workers', type=int)
    args = parser.parse_args()

    saved = json.loads((args.checkpoint.parent / 'config.json').read_text(encoding='utf-8'))
    namespace = {'__name__': '__fc_siam_diffspot__', 'REPO': REPO}
    for role, source in export_cells(json.loads(NOTEBOOK.read_text(encoding='utf-8'))):
        exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
        if role == 'config':
            apply_config(namespace, saved, data_root=args.data_root, num_workers=args.num_workers)

    import torch

    state = torch.load(args.checkpoint, map_location=namespace['device'], weights_only=True)
    namespace['net'].load_state_dict(state['model'])
    namespace['run_dir'] = args.checkpoint.parent
    print('checkpoint', args.checkpoint, 'pass', state['epoch'] + 1, flush=True)

    # Point the notebook's image loader at DiffSpot; rows carry only IDs and image paths
    dataset = args.dataset.resolve()
    namespace['DATA_ROOT'] = dataset
    rows = [json.loads(line) for line in (dataset / 'inference.jsonl').open()]
    raw = args.out.with_suffix('.full.jsonl')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    namespace['write_predictions'](rows, raw)

    # Keep the shared fields only, and check every pair appears exactly once
    lines = [json.loads(line) for line in raw.read_text().splitlines()]
    assert sorted(r['sample_id'] for r in lines) == sorted(r['sample_id'] for r in rows), 'IDs differ'
    with args.out.open('w') as out:
        for line in lines:
            out.write(json.dumps({k: line[k] for k in SHARED_FIELDS}) + '\n')
    boxes = sum(line['box'] is not None for line in lines)
    print(f'{len(lines)} predictions, {boxes} with a box, written to {args.out}')


if __name__ == '__main__':
    main()
