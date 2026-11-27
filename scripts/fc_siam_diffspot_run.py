"""Run the FC-Siam-diff DiffSpot benchmark: model -> Kimi K2.5 -> official DiffSpot scores.

Two modes, same command shape:

    python3 scripts/fc_siam_diffspot_run.py pilot runs/20261111-210302/ckpt_best.pt
    python3 scripts/fc_siam_diffspot_run.py full  runs/20261111-210302/ckpt_best.pt

``pilot`` runs the real chain on the team's 44-pair sample (about $0.10), checks every
file against the scorer's rules and writes ``pilot.md`` to read. ``full`` refuses to
start until a pilot has passed with the same checkpoint and the same code.

Every step is skipped when its output is complete, so rerunning the same command after
a crash, reboot or budget stop continues where it ended; nothing paid for is requested
again. A rerun also stops if any script, source or config file changed since the start,
because the frozen scorer would reject the mix.
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
sys.path.insert(0, str(REPO / 'src'))

from diffspot_judge import complete
from diffspot_kimi import NO_CHANGE
from openrouter import BUDGET_EXIT, FATAL_EXIT, load_key, remaining_credit, resume
from regression_lens.data.diffspot import load_prepared, read_json
from regression_lens.evaluation import diffspot as scoring
from regression_lens.evaluation._diffspot_official.metrics import aggregate
from regression_lens.evaluation.metrics import box_iou

PY = sys.executable
RUN_ID = 'selected-fc-siam'
SPEC = REPO / 'configs/diffspot.fc-siam.json'
CODE = ('scripts', 'src/regression_lens', 'configs')
# API transport only (retries, hosts); changing these doesn't change any answer or score, so it can't block a resume
TRANSPORT = ('scripts/openrouter.py', 'scripts/diffspot_judge.py')
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


def code_hash() -> str:
    digest = hashlib.sha256()
    for folder in CODE:
        for path in sorted((REPO / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and str(path.relative_to(REPO)) not in TRANSPORT:
                digest.update(str(path.relative_to(REPO)).encode() + path.read_bytes())
    return digest.hexdigest()


def same_code(out: Path, checkpoint_sha: str) -> dict:
    # The first run records the code and checkpoint; every rerun must match them
    record, current = out / 'code.json', dict(code_sha256=code_hash(), checkpoint_sha256=checkpoint_sha)
    if record.exists():
        if json.loads(record.read_text()) != current:
            raise SystemExit(f'Code or checkpoint changed since this run started ({record}). '
                             'Restore the original checkout, or start over with a new --out.')
    else:
        record.write_text(json.dumps(current, indent=2) + '\n')
    return current


def prepare_and_freeze(out: Path, dataset: Path, checkpoint_sha: str) -> Path:
    step('Prepare DiffSpot')
    if (dataset / 'manifest.json').exists():
        print('already prepared')
    else:
        if dataset.exists():
            shutil.rmtree(dataset)
        run(REPO / 'scripts/diffspot.py', 'prepare', '--out', dataset, retries=3)

    step('Freeze settings')
    frozen = out / 'frozen.json'
    if frozen.exists():
        print('already frozen')
    else:
        spec = json.loads(SPEC.read_text())
        spec['runs'][0]['checkpoint_sha256'] = checkpoint_sha
        spec['runs'][0]['validation_metrics_file'] = str((SPEC.parent / spec['runs'][0]['validation_metrics_file']).resolve())
        spec['kimi']['prompt_file'] = str((SPEC.parent / spec['kimi']['prompt_file']).resolve())
        (out / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
        run(REPO / 'scripts/diffspot_freeze_single.py', '--dataset', dataset, '--spec', out / 'spec.json', '--out', frozen)
    return frozen


def predict(out: Path, dataset: Path, checkpoint: Path, data_root, sample=None) -> Path:
    step('FC-Siam-diff predictions (resumes by chunk)')
    base = out / 'base.jsonl'
    if base.exists():
        print('already predicted')
    else:
        extra = (['--data-root', data_root] if data_root else []) + (['--sample', sample] if sample else [])
        run(REPO / 'scripts/fc_siam_diffspot_predict.py', checkpoint, '--dataset', dataset, '--out', base, *extra)
    return base


def pilot(args, out: Path, dataset: Path, checkpoint: Path, checkpoint_sha: str, code: dict) -> None:
    frozen = prepare_and_freeze(out, dataset, checkpoint_sha)

    step('Team 44-pair sample')
    sample = out / 'sample'
    if fresh_dir(sample, 'sample.json'):
        run(REPO / 'scripts/diffspot_smoke.py', 'sample', '--dataset', dataset, '--n', '44', '--out', sample)
    chosen = read_json(sample / 'sample.json')['sample_ids']
    _, every = load_prepared(dataset)
    annotations = [a for a in every if a['sample_id'] in set(chosen)]

    base = predict(out, dataset, checkpoint, args.data_root, sample)
    problems = []
    run_record = read_json(frozen)['runs'][0]
    try:
        indexed = scoring.validate_base(lines(base), annotations, run_record)
    except ValueError as error:
        raise SystemExit(f'FC-Siam-diff output fails the Test 1 format check: {error}')

    step('Kimi K2.5 on the sample')
    run(REPO / 'scripts/diffspot_kimi.py', '--dataset', dataset, '--predictions', base, '--out', out / 'kimi',
        '--sample', sample, '--max-cost', '0.5', retries=3)
    official = out / 'kimi.jsonl'
    run(REPO / 'scripts/diffspot_package.py', '--dataset', dataset, '--frozen', frozen, '--run', RUN_ID,
        '--responses', out / 'kimi/responses.jsonl', '--out', official, '--sample', sample)
    model, version = f"{RUN_ID}+{read_json(frozen)['kimi']['model']}", read_json(frozen)['kimi']['prompt_version']
    try:
        predictions = scoring.validate_official(lines(official), annotations, model, version)
    except ValueError as error:
        raise SystemExit(f'Official file fails the DiffSpot format check: {error}')
    # The same decision rules score-vlm enforces
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    decisions = {row['sample_id']: row for row in lines(out / 'kimi/verification.jsonl')}
    for a in annotations:
        sid, item = a['sample_id'], decisions.get(a['sample_id'])
        if item is None or item['decision'] not in ('accept', 'reject', 'unsure') \
                or item['source_prediction_sha256'] != base_sha \
                or item['description'] != predictions[sid]['prediction'] \
                or (item['decision'] == 'accept' and (indexed[sid]['box'] is None or item['box'] != indexed[sid]['box'])) \
                or (item['decision'] != 'accept' and (item['box'] is not None or item['description'] != NO_CHANGE)):
            problems.append(f'decision row for {sid} breaks the score-vlm rules')

    step('Official judge on the sample')
    requests = out / 'judge_requests.jsonl'
    with requests.open('w') as handle:
        # Same request rows as diffspot.py judge-inputs builds for the full run
        for a in annotations:
            handle.write(json.dumps(dict(id=a['sample_id'], split=a['split'], operator=a['operator'],
                gt_description=a['gt_description'] if a['split'] != 'no_diff' else
                    'No changes were applied; the two screenshots are independent renderings of the same HTML.',
                gt_mutations=a['gt_mutations'], prediction=predictions[a['sample_id']]['prediction'])) + '\n')
    run(REPO / 'scripts/diffspot_judge.py', '--requests', requests, '--out', out / 'judged.jsonl',
        '--max-cost', '0.2', retries=3)
    judged = {row['id']: row for row in lines(out / 'judged.jsonl')}
    for a in annotations:
        expected = 0 if a['split'] == 'no_diff' else len(a['gt_mutations'])
        if a['sample_id'] not in judged or not complete(judged[a['sample_id']]['judge_label'], expected):
            problems.append(f"judge row for {a['sample_id']} would be rejected by score-vlm")
    report = aggregate([dict(id=a['sample_id'], split=a['split'], operator=a['operator'],
                             is_true_positive=judged[a['sample_id']]['is_true_positive'],
                             is_true_negative=judged[a['sample_id']]['is_true_negative'])
                        for a in annotations if a['sample_id'] in judged]).to_dict()
    write_pilot(out, annotations, indexed, decisions, judged, report, problems, code)


def write_pilot(out, annotations, indexed, decisions, judged, report, problems, code) -> None:
    responses = {row['sample_id']: row for row in lines(out / 'kimi/responses.jsonl')}
    called = [r for r in responses.values() if r['called']]
    kimi_cost = sum(r['usage'].get('cost', 0) for r in called)
    judge_cost = sum(r.get('cost', 0) for r in judged.values())
    changed = [a for a in annotations if a['split'] != 'no_diff']
    boxed = [a for a in changed if indexed[a['sample_id']]['box'] is not None]
    ious = [box_iou(indexed[a['sample_id']]['box'], a['target_box']) for a in boxed]
    per_call = kimi_cost / max(1, len(called))
    full_calls = 3900 * len(boxed) / max(1, len(changed))
    projected = full_calls * per_call + 4400 * judge_cost / max(1, len(judged))
    checks = dict(passed=not problems, problems=problems, **code)
    tally = [row['decision'] for row in decisions.values()]
    counts = ', '.join(f'{d} {tally.count(d)}' for d in ('accept', 'reject', 'unsure'))
    (out / 'pilot.json').write_text(json.dumps(dict(checks, projected_full_cost_usd=round(projected, 2)), indent=2) + '\n')

    md = [f"# FC-Siam-diff → Kimi → DiffSpot pilot ({len(annotations)} pairs)", '',
          f"**Format checks: {'PASSED' if not problems else 'FAILED'}**", *[f'- {p}' for p in problems], '',
          '| | |', '|---|---|',
          f'| Changed pairs with a FC-Siam-diff box | {len(boxed)} / {len(changed)} |',
          f"| No-change pairs with a box | {sum(indexed[a['sample_id']]['box'] is not None for a in annotations if a['split'] == 'no_diff')} |",
          f"| Median box IoU with DiffSpot's target element | {sorted(ious)[len(ious) // 2]:.2f} |" if ious else '| Median box IoU | — |',
          f'| Kimi decisions | {counts} |',
          f"| Found (judge: correct) on changed pairs | {report['n_tp']} / {len(changed)} |",
          f"| No-change pairs answered correctly | {report['n_tn']} / {len(annotations) - len(changed)} |",
          f'| Pilot cost | ${kimi_cost + judge_cost:.4f} (Kimi ${per_call:.4f} per call) |',
          f'| **Projected full-run cost** | **${projected:.2f}** |', '',
          'These numbers are a plumbing check on 44 pairs, not results; do not tune anything on them.', '',
          '| Split | Type | IoU | Kimi | Kimi description | DiffSpot answer | Judge |', '|---|---|---|---|---|---|---|']
    for a in annotations:
        sid, box = a['sample_id'], indexed[a['sample_id']]['box']
        iou = f"{box_iou(box, a['target_box']):.2f}" if box and a['target_box'] else '—'
        verdicts = ', '.join(m['verdict'] for m in judged.get(sid, {}).get('judge_label', {}).get('mutations', [])) or \
            ('hallucinated' if judged.get(sid, {}).get('judge_label', {}).get('hallucinations') else 'ok')
        cell = lambda text: (text or '').replace('|', '/').replace('\n', ' ')[:160]
        md.append(f"| {a['split']} | {a['operator'] or '—'} | {iou} | {decisions[sid]['decision']} | "
                  f"{cell(decisions[sid]['description'])} | {cell(a['gt_description'])} | {verdicts} |")
    (out / 'pilot.md').write_text('\n'.join(md) + '\n')
    step('Pilot done')
    print('\n'.join(md[:14]))
    print(f'\nRead {out / "pilot.md"} before the full run.')
    if problems:
        raise SystemExit('Pilot FAILED the format checks; do not start the full run.')


def full(args, out: Path, dataset: Path, checkpoint: Path, checkpoint_sha: str, code: dict) -> None:
    # The full run needs a passed pilot made with this exact code and checkpoint
    record = args.pilot_out.resolve() / 'pilot.json'
    if not record.exists():
        raise SystemExit(f'No pilot found at {record}; run the pilot first.')
    passed = json.loads(record.read_text())
    if not passed['passed'] or {k: passed[k] for k in code} != code:
        raise SystemExit('The pilot failed, or was run with different code or a different checkpoint; rerun the pilot.')

    frozen = prepare_and_freeze(out, dataset, checkpoint_sha)
    base = predict(out, dataset, checkpoint, args.data_root)

    step('Test 1: FC-Siam-diff alone')
    if fresh_dir(out / 'test1', 'metrics.json'):
        run(REPO / 'scripts/diffspot.py', 'score-base', '--dataset', dataset, '--frozen', frozen, '--run', RUN_ID,
            '--predictions', base, '--out', out / 'test1')
    print('written to', out / 'test1/metrics.json')

    # Check before spending: enough credit to finish
    rows = lines(base)
    with_box = sum(row['box'] is not None for row in rows)
    kimi_left = with_box - sum(row['called'] for row in resume(out / 'kimi/responses.jsonl'))
    judge_left = len(rows) - len(resume(out / 'judged.jsonl'))
    estimate = kimi_left * KIMI_PER_CALL + judge_left * JUDGE_PER_CALL
    credit = remaining_credit(load_key())
    print(f'{with_box} pairs have a box; {kimi_left} Kimi and {judge_left} judge calls left, about ${estimate:.2f} '
          f"(pilot projected ${passed['projected_full_cost_usd']:.2f}); account credit ${credit:.2f}")
    if credit < estimate * 1.1 and not args.skip_credit_check:
        raise SystemExit('Not enough credit to finish without stopping partway. Top up, then rerun.')

    step('Kimi K2.5 verification (resumes by pair)')
    if len(lines(out / 'kimi/verification.jsonl')) != len(rows):
        run(REPO / 'scripts/diffspot_kimi.py', '--dataset', dataset, '--predictions', base, '--out', out / 'kimi',
            '--max-cost', args.kimi_max_cost, retries=20)
    print('all pairs answered')

    step('Official prediction file')
    official = out / 'kimi.jsonl'
    if not official.exists():
        run(REPO / 'scripts/diffspot_package.py', '--dataset', dataset, '--frozen', frozen, '--run', RUN_ID,
            '--responses', out / 'kimi/responses.jsonl', '--out', official)

    step('Judge requests')
    if fresh_dir(out / 'judge-inputs', 'judge-metadata.json'):
        run(REPO / 'scripts/diffspot.py', 'judge-inputs', '--test', '2', '--dataset', dataset, '--frozen', frozen,
            '--run', RUN_ID, '--predictions', official, '--out', out / 'judge-inputs')

    step('Judge (resumes by pair)')
    run(REPO / 'scripts/diffspot_judge.py', '--requests', out / 'judge-inputs/judge_requests.jsonl',
        '--out', out / 'judged.jsonl', '--max-cost', args.judge_max_cost, retries=20)

    step('Test 2: FC-Siam-diff + Kimi, official DiffSpot scores')
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('pilot', 'full'))
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path, help='Design2Code data folder, for the notebook setup')
    parser.add_argument('--dataset', type=Path, default=REPO / 'data/external/diffspot-full')
    parser.add_argument('--out', type=Path, help='default outputs/diffspot/fc-siam (full) or fc-siam-pilot (pilot)')
    parser.add_argument('--pilot-out', type=Path, default=REPO / 'outputs/diffspot/fc-siam-pilot')
    parser.add_argument('--kimi-max-cost', type=float, default=10.0)
    parser.add_argument('--judge-max-cost', type=float, default=2.5)
    parser.add_argument('--skip-credit-check', action='store_true', help='start Kimi even if credit looks short')
    args = parser.parse_args()
    out = (args.out or (args.pilot_out if args.mode == 'pilot' else REPO / 'outputs/diffspot/fc-siam')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset, checkpoint = args.dataset.resolve(), args.checkpoint.resolve()
    load_key()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    code = same_code(out, checkpoint_sha)
    (pilot if args.mode == 'pilot' else full)(args, out, dataset, checkpoint, checkpoint_sha, code)


if __name__ == '__main__':
    main()
