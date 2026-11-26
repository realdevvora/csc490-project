"""Export FC-Siam checkpoints using the training notebook's own prediction recipe.

From the repo root: python3 scripts/export_fc_siam_predictions.py --help
No optimization, training spot-checks, or new run folders are executed.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / 'notebooks/fc_siam_diff/train_fc_siam_diff.ipynb'
sys.path.insert(0, str(REPO / 'src'))


def export_cells(notebook):
    """Fail closed if a code cell has not been assigned an export role."""
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        role = cell.get('metadata', {}).get('regression_lens_export')
        source = ''.join(cell['source'])
        if role == 'skip':
            continue
        if role not in {'config', 'setup', 'predict'}:
            raise ValueError('Notebook code cell needs a regression_lens_export role.')
        if role == 'predict':
            tree = ast.parse(source)
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == 'write_predictions']
            if len(functions) != 1:
                raise ValueError('Expected exactly one write_predictions definition.')
            source = ast.unparse(ast.Module(body=functions, type_ignores=[]))
        yield role, source


def apply_config(namespace, saved, *, data_root=None, split_file=None, num_workers=None):
    """Restore the saved recipe while retaining this checkout's runtime paths."""
    runtime = {'REPO', 'DATA_ROOT', 'SPLITS', 'RUNS_DIR', 'INDEX', 'RESUME'}
    for key, value in saved.items():
        if key.isupper() and key in namespace and key not in runtime:
            namespace[key] = value
    if data_root is not None:
        namespace['DATA_ROOT'] = data_root.resolve()
    if split_file is not None:
        namespace['SPLITS'] = split_file.resolve()
    if num_workers is not None:
        namespace['NUM_WORKERS'] = num_workers
    elif os.name == 'nt':
        namespace['NUM_WORKERS'] = 0  # notebook Dataset classes cannot use spawn
    namespace['INDEX'] = namespace['DATA_ROOT'] / 'master_dataset_with_benign.json'
    namespace['RESUME'] = None
    namespace['PAIRS_PER_LOAD'] = {
        'all_squares': 1, 'whole_page': namespace['BATCH_SIZE'],
    }.get(namespace['INPUT_MODE'], namespace['BATCH_SIZE'] // namespace['SQUARES_PER_PAIR'])


def newest_checkpoint():
    roots = (REPO / 'outputs/fc_siam_diff/runs', REPO / 'models/fc_siam_diff/runs',
             REPO / 'notebooks/fc_siam_diff/runs', Path.cwd() / 'runs')
    candidates = {p.resolve() for root in roots for p in root.glob('*/ckpt_best.pt')}
    if not candidates:
        raise FileNotFoundError('No ckpt_best.pt found. Pass a checkpoint path explicitly.')
    return max(candidates, key=lambda p: (p.parent.name, str(p)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', nargs='?', type=Path)
    parser.add_argument('out', nargs='?', type=Path, default=REPO / 'outputs/fc_siam_diff')
    parser.add_argument('--config', type=Path, help='Defaults to config.json beside checkpoint.')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--num-workers', type=int)
    args = parser.parse_args(argv)
    if args.num_workers is not None and args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    checkpoint = args.checkpoint.resolve() if args.checkpoint else newest_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = args.config or checkpoint.parent / 'config.json'
    saved = json.loads(config.read_text(encoding='utf-8'))
    cells = list(export_cells(json.loads(NOTEBOOK.read_text(encoding='utf-8'))))
    if sum(role == 'config' for role, _ in cells) != 1 or cells[0][0] != 'config':
        raise ValueError('Expected one config cell before setup.')
    namespace = {'__name__': '__fc_siam_export__', 'REPO': REPO}
    for role, source in cells:
        exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
        if role == 'config':
            apply_config(namespace, saved, data_root=args.data_root,
                         split_file=args.split_file, num_workers=args.num_workers)

    import torch

    state = torch.load(checkpoint, map_location=namespace['device'], weights_only=True)
    namespace['net'].load_state_dict(state['model'])
    namespace['run_dir'] = checkpoint.parent
    print('checkpoint', checkpoint, 'pass', state['epoch'] + 1, flush=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_path = out / 'fc_siam_diff_all.jsonl'
    namespace['write_predictions'](namespace['rows'], all_path)
    raw_lines = all_path.read_text(encoding='utf-8').splitlines()
    lines = {json.loads(line)['sample_id']: line for line in raw_lines}
    assert len(lines) == len(raw_lines), 'Duplicate exported sample IDs'
    assert sorted(lines) == sorted(r['sample_id'] for r in namespace['rows'])
    for name, rows in (('train', namespace['train_rows']), ('val', namespace['val_rows'])):
        ids = [r['sample_id'] for r in rows]
        if name == 'train':
            # These controls were omitted only during training, not scoring.
            ids += sorted(namespace['BLANK_RENDERS'] & lines.keys())
        (out / f'fc_siam_diff_{name}.jsonl').write_text(
            ''.join(lines[i] + '\n' for i in ids), encoding='utf-8')
        print(name, len(ids), 'predictions')
    print('all', len(lines), 'predictions; written to', out)


if __name__ == '__main__':
    main()
