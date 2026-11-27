"""Grade Kimi's descriptions with DiffSpot's official judge (gpt-oss-120b, prompt v2.0).

Reads the evaluator-only ``judge_requests.jsonl`` from ``diffspot.py judge-inputs``
and writes one judged row per pair for ``score-vlm``. Uses the official prompt,
reply parsing and reducer unchanged; only the API call goes through OpenRouter,
pinned to one full-precision host. Needs ``OPENROUTER_API_KEY`` (see diffspot_kimi.py).

    python3 scripts/diffspot_judge.py --requests outputs/diffspot/fc-siam-judge-inputs/judge_requests.jsonl \\
        --out outputs/diffspot/fc-siam-judged.jsonl
"""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from diffspot_kimi import call, cost, load_key, read_jsonl
from regression_lens.evaluation._diffspot_official.judge import (DEFAULT_MAX_TOKENS, DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE, _parse_judge_json, reduce_judge_label, render_prompt)

JUDGE_MODEL, JUDGE_PROVIDER, JUDGE_QUANTIZATION = 'openai/gpt-oss-120b', 'deepinfra', 'bf16'


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
    args = parser.parse_args()

    key = load_key()
    requests = read_jsonl(args.requests)
    done = {row['id'] for row in read_jsonl(args.out)} if args.out.exists() else set()
    spent = 0.0
    with args.out.open('a') as out:
        for index, item in enumerate(requests, 1):
            if item['id'] in done:
                continue
            body = dict(model=JUDGE_MODEL, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS,
                        reasoning={'effort': DEFAULT_REASONING_EFFORT}, usage={'include': True},
                        provider={'order': [JUDGE_PROVIDER], 'quantizations': [JUDGE_QUANTIZATION],
                                  'allow_fallbacks': False},
                        messages=[{'role': 'user', 'content': render_prompt(
                            gt_mutations=item['gt_mutations'], gt_description=item['gt_description'],
                            prediction=item['prediction'])}])
            # Like the official judge_item, retry up to three times; also retry a reply score-vlm would reject
            expected = 0 if item['split'] == 'no_diff' else len(item['gt_mutations'])
            for attempt in range(3):
                reply = call(key, body)
                spent += cost(reply.get('usage', {}))
                try:
                    label = _parse_judge_json(reply['choices'][0]['message'].get('content') or '')
                except ValueError:
                    label = None
                if complete(label, expected):
                    break
            else:
                raise SystemExit(f"judge reply for {item['id']} was incomplete after 3 tries")
            reduced = reduce_judge_label(label, is_no_diff=item['split'] == 'no_diff')
            out.write(json.dumps(dict(id=item['id'], split=item['split'], operator=item['operator'],
                                      judge_label=label, **reduced), ensure_ascii=False) + '\n')
            out.flush()
            print(f"{index}/{len(requests)} tp={reduced['is_true_positive']} tn={reduced['is_true_negative']}")
    print(f'judged {len(requests)} pairs; this session cost ${spent:.4f}')


if __name__ == '__main__':
    main()
