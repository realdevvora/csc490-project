"""Turn Kimi runner output into the official DiffSpot prediction file (six fields).

Runs after every Kimi call has finished: only then is each pair's ``split``
joined from ``annotations.jsonl``.

    python3 scripts/diffspot_package.py --dataset data/external/diffspot-full \\
        --frozen outputs/diffspot/fc-siam-frozen.json --run selected-fc-siam \\
        --responses outputs/diffspot/fc-siam-kimi/responses.jsonl --out outputs/diffspot/fc-siam-kimi.jsonl
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from regression_lens.data.diffspot import load_prepared, read_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('dataset', 'frozen', 'responses', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--sample', type=Path, help='saved sample folder; only its pairs are packaged')
    args = parser.parse_args()

    _, annotations = load_prepared(args.dataset)
    if args.sample:
        chosen = set(read_json(args.sample / 'sample.json')['sample_ids'])
        annotations = [a for a in annotations if a['sample_id'] in chosen]
    kimi = read_json(args.frozen)['kimi']
    responses = {row['sample_id']: row for row in read_jsonl(args.responses)}
    missing = [a['sample_id'] for a in annotations if a['sample_id'] not in responses]
    if missing:
        raise SystemExit(f'{len(missing)} pairs have no Kimi response yet, e.g. {missing[0]}')
    with args.out.open('w') as out:
        for a in annotations:
            row = responses[a['sample_id']]
            out.write(json.dumps(dict(id=a['sample_id'], split=a['split'], model=f"{args.run}+{kimi['model']}",
                                      prompt_version=kimi['prompt_version'], raw_response=row['raw_response'],
                                      prediction=row['description']), ensure_ascii=False) + '\n')
    print(f'wrote {len(annotations)} official predictions to {args.out}')


if __name__ == '__main__':
    main()
