"""Run the whole FC-Siam-diff DiffSpot benchmark (Test 1 and Test 2) in one command.

Every step is skipped when its output is already complete, so rerunning the same
command after a crash, a reboot or a budget stop continues from where it ended.
API steps retry on their own; nothing already paid for is requested again.

    python3 scripts/fc_siam_diffspot_run.py runs/20261111-210302/ckpt_best.pt \\
        [--data-root data/processed/d2c_v3] [--out outputs/diffspot/fc-siam] \\
        [--kimi-max-cost 9] [--judge-max-cost 2]

Steps: prepare DiffSpot, freeze, FC-Siam-diff predictions, Test 1, Kimi,
official file, judge requests, judge, Test 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from openrouter import BUDGET_EXIT, FATAL_EXIT, load_key, remaining_credit, resume

PY = sys.executable
RUN_ID = 'selected-fc-siam'
SPEC = REPO / 'configs/diffspot.fc-siam.json'
KIMI_PER_CALL, JUDGE_PER_CALL = 0.0021, 0.00035  # measured on the 44-pair sample, USD


def step(name: str) -> None:
    print(f'\n=== {name} ({time.strftime("%H:%M:%S")})', flush=True)


def run(*args, retries: int = 0) -> None:
    # API steps are retried after a pause; a budget stop or a refused request ends the whole run
    for attempt in range(retries + 1):
        code = subprocess.call([PY, *map(str, args)], cwd=REPO)
        if code == 0:
            return
        if code == BUDGET_EXIT:
            raise SystemExit('Stopped by the spending cap or account credit. Top up or raise the cap, '
                             'then rerun this same command to continue.')
        if attempt < retries and code != FATAL_EXIT:
            print(f'step failed (exit {code}); retrying in 2 minutes ({attempt + 1}/{retries})', flush=True)
            time.sleep(120)
    raise SystemExit(f'step failed (exit {code}); rerun this same command to continue')


def fresh_dir(path: Path, marker: str) -> bool:
    # True when the step must run; a half-written folder from a crash is cleared first
    if (path / marker).exists():
        return False
    if path.exists():
        shutil.rmtree(path)
    return True


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path, help='Design2Code data folder, for the notebook setup')
    parser.add_argument('--dataset', type=Path, default=REPO / 'data/external/diffspot-full')
    parser.add_argument('--out', type=Path, default=REPO / 'outputs/diffspot/fc-siam')
    parser.add_argument('--kimi-max-cost', type=float, default=9.0)
    parser.add_argument('--judge-max-cost', type=float, default=2.0)
    parser.add_argument('--skip-credit-check', action='store_true', help='start Kimi even if credit looks short')
    args = parser.parse_args()
    out, dataset, checkpoint = args.out.resolve(), args.dataset.resolve(), args.checkpoint.resolve()
    out.mkdir(parents=True, exist_ok=True)
    load_key()

    step('1. Prepare DiffSpot')
    if (dataset / 'manifest.json').exists():
        print('already prepared')
    else:
        if dataset.exists():
            shutil.rmtree(dataset)
        run(REPO / 'scripts/diffspot.py', 'prepare', '--out', dataset, retries=3)

    step('2. Freeze settings')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    frozen = out / 'frozen.json'
    if frozen.exists():
        if json.loads(frozen.read_text())['runs'][0]['checkpoint_sha256'] != checkpoint_sha:
            raise SystemExit(f'{frozen} was frozen for a different checkpoint; use a new --out')
        print('already frozen')
    else:
        spec = json.loads(SPEC.read_text())
        spec['runs'][0]['checkpoint_sha256'] = checkpoint_sha
        spec['runs'][0]['validation_metrics_file'] = str((SPEC.parent / spec['runs'][0]['validation_metrics_file']).resolve())
        spec['kimi']['prompt_file'] = str((SPEC.parent / spec['kimi']['prompt_file']).resolve())
        (out / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
        run(REPO / 'scripts/diffspot_freeze_single.py', '--dataset', dataset, '--spec', out / 'spec.json', '--out', frozen)

    step('3. FC-Siam-diff predictions (resumes by chunk)')
    base = out / 'base.jsonl'
    if base.exists():
        print('already predicted')
    else:
        extra = ['--data-root', args.data_root] if args.data_root else []
        run(REPO / 'scripts/fc_siam_diffspot_predict.py', checkpoint, '--dataset', dataset, '--out', base, *extra)

    step('4. Test 1: FC-Siam-diff alone')
    if fresh_dir(out / 'test1', 'metrics.json'):
        run(REPO / 'scripts/diffspot.py', 'score-base', '--dataset', dataset, '--frozen', frozen, '--run', RUN_ID,
            '--predictions', base, '--out', out / 'test1')
    print('written to', out / 'test1/metrics.json')

    # Check before spending: there must be boxes to verify, and enough credit to finish
    with_box = sum(row['box'] is not None for row in lines(base))
    if not with_box:
        raise SystemExit('FC-Siam-diff proposed no boxes at all; check the predictions before calling Kimi')
    answered = resume(out / 'kimi/responses.jsonl')
    kimi_left = with_box - sum(row['called'] for row in answered)
    judged = len(resume(out / 'judged.jsonl'))
    estimate = kimi_left * KIMI_PER_CALL + (len(lines(base)) - judged) * JUDGE_PER_CALL
    credit = remaining_credit(load_key())
    print(f'{with_box} pairs have a box; {kimi_left} Kimi calls and {len(lines(base)) - judged} judge calls left, '
          f'about ${estimate:.2f}; account credit ${credit:.2f}')
    if credit < estimate * 1.1 and not args.skip_credit_check:
        raise SystemExit('Not enough credit to finish without stopping partway. Top up, then rerun.')

    step('5. Kimi K2.5 verification (resumes by pair)')
    if len(lines(out / 'kimi/verification.jsonl')) != len(lines(base)):
        run(REPO / 'scripts/diffspot_kimi.py', '--dataset', dataset, '--predictions', base, '--out', out / 'kimi',
            '--max-cost', args.kimi_max_cost, retries=20)
    print('all pairs answered')

    step('6. Official prediction file')
    official = out / 'kimi.jsonl'
    if not official.exists():
        run(REPO / 'scripts/diffspot_package.py', '--dataset', dataset, '--frozen', frozen, '--run', RUN_ID,
            '--responses', out / 'kimi/responses.jsonl', '--out', official)

    step('7. Judge requests')
    if fresh_dir(out / 'judge-inputs', 'judge-metadata.json'):
        run(REPO / 'scripts/diffspot.py', 'judge-inputs', '--test', '2', '--dataset', dataset, '--frozen', frozen,
            '--run', RUN_ID, '--predictions', official, '--out', out / 'judge-inputs')

    step('8. Judge (resumes by pair)')
    run(REPO / 'scripts/diffspot_judge.py', '--requests', out / 'judge-inputs/judge_requests.jsonl',
        '--out', out / 'judged.jsonl', '--max-cost', args.judge_max_cost, retries=20)

    step('9. Test 2: FC-Siam-diff + Kimi, official DiffSpot scores')
    if fresh_dir(out / 'test2', 'metrics.json'):
        run(REPO / 'scripts/diffspot.py', 'score-vlm', '--dataset', dataset, '--frozen', frozen, '--run', RUN_ID,
            '--base-predictions', base, '--predictions', official, '--decisions', out / 'kimi/verification.jsonl',
            '--judged', out / 'judged.jsonl', '--judge-metadata', out / 'judge-inputs/judge-metadata.json',
            '--out', out / 'test2')

    report = json.loads((out / 'test2/official_scores.json').read_text())['report']
    spent = (json.loads((out / 'kimi/run.json').read_text())['cost_usd']
             + sum(row.get('cost', 0) for row in resume(out / 'judged.jsonl')))
    step('Done')
    print(json.dumps({k: report.get(k) for k in ('overall_accuracy', 'diff_overall_recall', 'no_diff_specificity')}, indent=2))
    print('by difficulty:', {tier: value.get('recall') for tier, value in report.get('by_tier', {}).items()})
    print(f'API cost ${spent:.2f}. Results in {out}/test1 and {out}/test2')


if __name__ == '__main__':
    main()
