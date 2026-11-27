"""Freeze one selected configuration for DiffSpot, for a single-model round.

The team's ``diffspot.py freeze`` requires all four configurations. This writes
the same frozen file for one run, so ``score-base``, ``judge-inputs`` and
``score-vlm`` work for that run alone. The spec has the same shape as
``configs/diffspot.example.json`` with exactly one entry in ``runs``.

    python3 scripts/diffspot_freeze_single.py --dataset data/external/diffspot-full \\
        --spec outputs/diffspot/fc-siam-spec.json --out outputs/diffspot/fc-siam-frozen.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from regression_lens.data.diffspot import digest, load_prepared, read_json, read_jsonl
from regression_lens.evaluation import diffspot as scoring


def freeze_single(dataset: Path, specification: Path, output: Path) -> dict:
    # Same checks and output fields as diffspot.py freeze, for one run instead of four
    manifest, _ = load_prepared(dataset)
    if manifest['preliminary']:
        raise ValueError('Smoke data cannot be frozen')
    scoring.verify_official_sources()
    for image in read_jsonl(dataset / 'inference.jsonl'):
        for side in ('reference', 'candidate'):
            if digest(dataset / image[side + '_path']) != image[side + '_sha256']:
                raise ValueError('Prepared screenshots changed before freezing')
    spec = read_json(specification)
    if spec.get('selection_dataset') != 'design2code_val':
        raise ValueError('Selection must be declared design2code_val, never DiffSpot')
    if len(spec.get('runs', [])) != 1:
        raise ValueError('Give exactly one run')
    row, base = spec['runs'][0], specification.resolve().parent
    for key in ('id', 'family', 'seed_policy', 'preprocessing', 'box_rule'):
        if not row.get(key) or 'REPLACE' in str(row[key]):
            raise ValueError(f'Run requires {key}')
    checkpoint = row.get('checkpoint_sha256', '')
    if len(checkpoint) != 64 or any(c not in '0123456789abcdef' for c in checkpoint):
        raise ValueError('Supply the real checkpoint SHA-256')
    metrics_file = base / row['validation_metrics_file']
    metrics = read_json(metrics_file)
    if not scoring.finite(metrics.get('threshold')) or metrics.get('n') != 1033:
        raise ValueError('Provide shared Design2Code validation metrics for all 1,033 pairs')
    run = dict(id=row['id'], family=row['family'], checkpoint_sha256=checkpoint,
               checkpoint_fingerprint_source='model owner; weights not read by this dataset tool',
               seed_policy=row['seed_policy'], preprocessing=row['preprocessing'], box_rule=row['box_rule'],
               supports_type=bool(row.get('supports_type', False)), threshold=metrics['threshold'],
               validation_metrics_sha256=digest(metrics_file),
               base_prompt_version=row.get('base_prompt_version', 'base-v1'))
    kimi = spec['kimi']
    if 'k2.5' not in kimi.get('model', '').lower():
        raise ValueError('This benchmark fixes Kimi K2.5')
    settings = dict(model=kimi['model'], prompt_version=kimi['prompt_version'],
                    prompt_sha256=digest(base / kimi['prompt_file']), temperature=0, max_tokens=16384,
                    max_regions=1, crop_padding_px=32, overview_width_px=1280,
                    coordinate_refinement=False, no_diff_response=scoring.NO_DIFF_RESPONSE,
                    **{k: v for k, v in kimi.items() if k not in ('model', 'prompt_version', 'prompt_file')})
    value = dict(format_version=1, selection_dataset='design2code_val', runs=[run], kimi=settings,
                 dataset_manifest_sha256=digest(dataset / 'manifest.json'), preliminary=False,
                 official_provenance_sha256=digest(scoring.OFFICIAL_ROOT / 'PROVENANCE.json'),
                 design2code_split_sha256=digest(REPO / 'data/splits/splits_85_15.json'),
                 scorer_sha256=digest(Path(scoring.__file__)), source_specification_sha256=digest(specification),
                 policy='No DiffSpot training, threshold/prompt/checkpoint tuning or per-model VLM tuning',
                 round='single-model')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation, so a frozen round is never overwritten
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = freeze_single(args.dataset, args.spec, args.out)
    print(f"froze {value['runs'][0]['id']} (threshold {value['runs'][0]['threshold']:.4f}) to {args.out}")


if __name__ == '__main__':
    main()
