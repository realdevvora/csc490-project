"""Grade Kimi's descriptions with DiffSpot's official judge (gpt-oss-120b, prompt v2.0).

Reads the evaluator-only ``judge_requests.jsonl`` from ``diffspot.py judge-inputs``
and appends one judged row per pair for ``score-vlm``. Uses the official prompt,
reply parsing and reducer unchanged; only the API call goes through OpenRouter,
limited to a fixed list of hosts that serve the full model. Rerunning continues where it stopped; exit
code 3 means the spending cap or the account credit stopped it.

    python3 scripts/diffspot_judge.py --requests outputs/diffspot/fc-siam-judge-inputs/judge_requests.jsonl \\
        --out outputs/diffspot/fc-siam-judged.jsonl [--max-cost 2] [--workers 8]
"""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from openrouter import BUDGET_EXIT, BudgetStop, call, cost, load_key, resume, run_all
from regression_lens.evaluation._diffspot_official.judge import (DEFAULT_MAX_TOKENS, DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE, _parse_judge_json, reduce_judge_label, render_prompt)

JUDGE_MODEL = 'openai/gpt-oss-120b'
# Hosts tried in this order, all serving the full model (bf16, or its native fp4); no other host is used
JUDGE_PROVIDERS = ['deepinfra', 'crusoe', 'google-vertex', 'amazon-bedrock', 'together', 'nebius', 'baseten']


def complete(label, expected: int) -> bool:
    # One verdict per true change and a hallucination list, as score-vlm requires
    return (isinstance(label, dict) and isinstance(label.get('mutations'), list)
            and isinstance(label.get('hallucinations'), list) and len(label['mutations']) == expected
            and all(isinstance(m, dict) and m.get('verdict') in ('correct', 'partial', 'missed')
                    for m in label['mutations']))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-cost', type=float, default=2.0, help='stop starting calls once this many USD are spent')
    parser.add_argument('--workers', type=int, default=4, help='calls in flight at once')
    args = parser.parse_args()

    requests = [json.loads(line) for line in args.requests.open() if line.strip()]
    done = resume(args.out)
    finished = {row['id'] for row in done}
    spent = sum(row.get('cost', 0) for row in done)
    todo = [item for item in requests if item['id'] not in finished]
    print(f'{len(finished)} of {len(requests)} already judged (${spent:.4f}); {len(todo)} to go', flush=True)
    key = load_key() if todo else ''

    def work(item: dict) -> tuple[dict, float]:
        body = dict(model=JUDGE_MODEL, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS,
                    reasoning={'effort': DEFAULT_REASONING_EFFORT}, usage={'include': True},
                    provider={'order': JUDGE_PROVIDERS, 'allow_fallbacks': False},
                    messages=[{'role': 'user', 'content': render_prompt(
                        gt_mutations=item['gt_mutations'], gt_description=item['gt_description'],
                        prediction=item['prediction'])}])
        # Like the official judge_item, retry up to three times; also retry a reply score-vlm would reject
        expected = 0 if item['split'] == 'no_diff' else len(item['gt_mutations'])
        charged = 0.0
        for _ in range(3):
            reply = call(key, body)
            charged += cost(reply.get('usage'))
            try:
                label = _parse_judge_json(reply['choices'][0]['message'].get('content') or '')
            except ValueError:
                label = None
            if complete(label, expected):
                break
        else:
            raise RuntimeError(f"judge reply for {item['id']} was incomplete after 3 tries")
        reduced = reduce_judge_label(label, is_no_diff=item['split'] == 'no_diff')
        print(f"  {item['split']:7s} tp={reduced['is_true_positive']} tn={reduced['is_true_negative']}", flush=True)
        row = dict(id=item['id'], split=item['split'], operator=item['operator'], judge_label=label, **reduced)
        return dict(row, cost=charged, provider=reply.get('provider')), charged

    try:
        spent = run_all(todo, work, args.out, spent, args.max_cost, args.workers)
    except BudgetStop as stop:
        print(stop)
        raise SystemExit(BUDGET_EXIT)
    print(f'judged all {len(requests)} pairs; judge cost ${spent:.4f}')


if __name__ == '__main__':
    main()
