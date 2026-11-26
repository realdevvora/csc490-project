"""Prepare and score frozen DiffSpot experiments. No model or paid API calls.

Run prepare/freeze/score-base/score-vlm/judge-inputs/report --help for details.
"""
from __future__ import annotations

import argparse
import csv
from io import BytesIO
from itertools import islice
from pathlib import Path
import sys

from regression_lens.data.diffspot import (digest, fresh, load_prepared, prepare,
    read_json, read_jsonl, write_json, write_jsonl)
from regression_lens.evaluation import diffspot as scoring

REPO = Path(__file__).resolve().parents[1]


def image_rows(rows, source_dir=None):
    from PIL import Image
    for item in rows:
        row = dict(item)
        for field in ('image_before', 'image_after'):
            value = row[field]
            if isinstance(value, str):
                value = Image.open(Path(source_dir or '.')/value)
            elif isinstance(value, dict):
                value = Image.open(BytesIO(value['bytes'])) if value.get('bytes') is not None else Image.open(value['path'])
            row[field] = value
        yield row


def prepare_command(args):
    if args.system_trust:
        import truststore
        truststore.inject_into_ssl()
    if args.limit_pairs is not None and args.limit_pairs < 1:
        raise ValueError('limit-pairs must be positive')
    if args.source_jsonl:
        source = Path(args.source_jsonl)
        rows = image_rows(read_jsonl(source), source.parent)
        revision = 'local-jsonl:'+digest(source)
    else:
        import os
        cache_root = Path(args.out).resolve().parent/'hf_cache'
        # Set before importing HF libraries so metadata caches also stay off
        # the lab home quota. Respect an explicitly configured HF_HOME.
        os.environ.setdefault('HF_HOME', str(cache_root/'home'))
        try:
            from datasets import Image, load_dataset
            from huggingface_hub import HfApi
        except ImportError as error:
            raise ValueError('Install the diffspot optional dependency group to download the release') from error
        revision = HfApi(token=False).dataset_info('tencent/DiffSpot', revision=args.revision).sha
        rows = load_dataset('tencent/DiffSpot', split='test', revision=revision, streaming=True,
                            cache_dir=str(cache_root), token=False)
        # Decode with Pillow ourselves, avoiding datasets-version-specific image decoders.
        rows = rows.cast_column('image_before', Image(decode=False)).cast_column('image_after', Image(decode=False))
        rows = image_rows(rows)
    if args.limit_pairs:
        rows = islice(rows, args.limit_pairs)
    manifest = prepare(rows, args.out, revision, preliminary=args.limit_pairs is not None)
    print(f"Prepared {manifest['n']} pairs; preliminary={manifest['preliminary']}; revision={revision}")


def freeze(dataset, specification, output, allow_preliminary=False):
    manifest, _ = load_prepared(dataset)
    if manifest['preliminary'] and not allow_preliminary:
        raise ValueError('Smoke data cannot be frozen as a full benchmark; use --allow-preliminary for plumbing tests')
    scoring.verify_official_sources()
    for image in read_jsonl(Path(dataset)/'inference.jsonl'):
        for side in ('reference', 'candidate'):
            if digest(Path(dataset)/image[side+'_path']) != image[side+'_sha256']:
                raise ValueError('Prepared screenshots changed before freezing')
    spec = read_json(specification)
    if spec.get('selection_dataset') != 'design2code_val':
        raise ValueError('Selection must be declared design2code_val, never DiffSpot')
    runs = spec.get('runs', [])
    ids = [r.get('id') for r in runs]
    if len(runs) != 4 or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != 4:
        raise ValueError('Freeze exactly four uniquely named selected configurations')
    base = Path(specification).resolve().parent
    locked = []
    for row in runs:
        for key in ('family', 'seed_policy', 'preprocessing', 'box_rule'):
            if not row.get(key) or 'REPLACE' in str(row[key]):
                raise ValueError(f'Run requires {key}')
        checkpoint = row.get('checkpoint_sha256', '')
        if len(checkpoint) != 64 or any(c not in '0123456789abcdef' for c in checkpoint):
            raise ValueError('Supply a real checkpoint SHA256 from the model owner')
        metrics_file = base/row['validation_metrics_file']
        metrics = read_json(metrics_file)
        threshold = metrics.get('threshold')
        if not scoring.finite(threshold) or metrics.get('n') != 1033:
            raise ValueError('Provide shared Design2Code validation metrics for all 1,033 pairs and a finite threshold')
        if not isinstance(row.get('supports_type', False), bool):
            raise ValueError('supports_type must be boolean')
        locked.append(dict(id=row['id'], family=row['family'], checkpoint_sha256=checkpoint,
            checkpoint_fingerprint_source='model owner; weights not read by this dataset tool',
            seed_policy=row['seed_policy'], preprocessing=row['preprocessing'], box_rule=row['box_rule'],
            supports_type=row.get('supports_type', False), threshold=threshold,
            validation_metrics_sha256=digest(metrics_file), base_prompt_version=row.get('base_prompt_version', 'base-v1')))
    kimi = spec['kimi']
    if not all(kimi.get(k) for k in ('model', 'prompt_version', 'prompt_file')):
        raise ValueError('Shared Kimi identity and prompt must be supplied')
    if 'kimi' not in kimi['model'].lower() or 'k2.5' not in kimi['model'].lower():
        raise ValueError('This benchmark fixes Kimi K2.5 for every pipeline')
    settings = dict(model=kimi['model'], prompt_version=kimi['prompt_version'],
                    prompt_sha256=digest(base/kimi['prompt_file']), temperature=0, max_tokens=16384,
                    max_regions=1, crop_padding_px=32, overview_width_px=1280,
                    coordinate_refinement=False, no_diff_response=scoring.NO_DIFF_RESPONSE)
    value = dict(format_version=1, selection_dataset='design2code_val', runs=locked, kimi=settings,
        dataset_manifest_sha256=digest(Path(dataset)/'manifest.json'), preliminary=manifest['preliminary'],
        official_provenance_sha256=digest(scoring.OFFICIAL_ROOT/'PROVENANCE.json'),
        design2code_split_sha256=digest(REPO/'data/splits/splits_85_15.json'),
        scorer_sha256=digest(Path(scoring.__file__)), source_specification_sha256=digest(specification),
        policy='No DiffSpot training, threshold/prompt/checkpoint tuning or per-model VLM tuning')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves a frozen round.
    with output.open('x', encoding='utf-8') as stream:
        import json
        stream.write(json.dumps(value, indent=2)+'\n')
    return value


def judge_inputs(args):
    _, annotations, frozen, run = scoring.get_run(args.dataset, args.frozen, args.run)
    model = args.run if args.test == 1 else args.run+'+'+frozen['kimi']['model']
    version = run['base_prompt_version'] if args.test == 1 else frozen['kimi']['prompt_version']
    indexed = scoring.validate_official(read_jsonl(args.predictions), annotations, model, version)
    out = fresh(args.out)
    context = scoring.judge_context(args.predictions, args.dataset, args.frozen, model, version)
    # This file is evaluator-only. It must NEVER be sent to Kimi inference.
    write_jsonl(out/'judge_requests.jsonl', [dict(id=a['sample_id'], split=a['split'], operator=a['operator'],
        gt_description=a['gt_description'] if a['split'] != 'no_diff' else
          'No changes were applied; the two screenshots are independent renderings of the same HTML.',
        gt_mutations=a['gt_mutations'], prediction=indexed[a['sample_id']]['prediction']) for a in annotations])
    write_json(out/'judge-metadata.json', context)
    print('Wrote EVALUATOR-ONLY judge inputs. No API calls were made.')


def report(paths, frozen_path, output):
    import json
    frozen = read_json(frozen_path)
    results = [read_json(Path(p)/'metrics.json' if Path(p).is_dir() else p) for p in paths]
    seen, rows = set(), []
    registered = {r['id'] for r in frozen['runs']}
    for r in results:
        key = (r['run'], r['test'])
        if key in seen or r['run'] not in registered or r['test'] not in (1, 2):
            raise ValueError('Duplicate or unregistered comparison result')
        if r['frozen_sha256'] != digest(frozen_path):
            raise ValueError('Cannot compare different frozen benchmark rounds')
        seen.add(key)
        m = r['official_metrics']
        loc = r['supplementary']['localization'] if r['test'] == 1 else r['supplementary']['final_localization']
        rows.append(dict(run=r['run'], test=r['test'], official_status=r['official_status'],
            easy_recall=m['by_tier'].get('easy',{}).get('recall'),
            medium_recall=m['by_tier'].get('medium',{}).get('recall'),
            hard_recall=m['by_tier'].get('hard',{}).get('recall'),
            diff_overall_recall=m['diff_overall_recall'], no_diff_specificity=m['no_diff_specificity'],
            overall_accuracy=m['overall_accuracy'], n_total=m['n_total'], n_has_diff=m['n_has_diff'],
            n_no_diff=m['n_no_diff'], mean_iou_all_changed=loc['mean_iou_all_changed'], n_box=loc['n_box']))
        detector_report = r['supplementary'].get('detector_only_report')
        for tier, name in [('easy','detector_easy_recall'),('medium','detector_medium_recall'),('hard','detector_hard_recall')]:
            rows[-1][name] = detector_report['by_tier'].get(tier,{}).get('recall') if detector_report else None
        for key in ('diff_overall_recall','no_diff_specificity','overall_accuracy'):
            rows[-1]['detector_'+key] = detector_report[key] if detector_report else None
    paired = []
    for run in sorted(registered):
        pair = [r for r in results if r['run'] == run]
        if len(pair) == 2:
            a = next(r for r in pair if r['test'] == 1); b = next(r for r in pair if r['test'] == 2)
            if a['predictions_sha256'] != b['base_predictions_sha256']:
                raise ValueError('Test 2 was not built from the compared Test 1 predictions')
            delta = {key: b['official_metrics'][key]-a['official_metrics'][key]
                     if a['official_metrics'][key] is not None else None
                     for key in ('overall_accuracy','diff_overall_recall','no_diff_specificity')}
            paired.append(dict(run=run, official_deltas=delta))
    complete = len(seen) == 8 and not frozen['preliminary'] and not any(r['preliminary'] for r in results)
    out = fresh(output)
    write_json(out/'comparison.json', dict(complete_four_pipeline_comparison=complete, results=results, paired=paired))
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else []);writer.writeheader();writer.writerows(rows)
    columns = ['run','test','easy_recall','medium_recall','hard_recall','diff_overall_recall','no_diff_specificity','overall_accuracy','mean_iou_all_changed','n_box']
    def fmt(x): return '—' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
    text = ['# DiffSpot frozen benchmark comparison', '',
            'Complete four-pipeline comparison: '+str(complete), '',
            '| '+' | '.join(columns)+' |','|'+'---|'*len(columns)]
    text += ['| '+' | '.join(fmt(r[k]) for k in columns)+' |' for r in rows]
    text += ['', 'Dashes are unavailable description scores, never binary detector scores.',
             'Localization is against target-element boxes. Original proposal coordinates are fixed.',
             'Published Kimi-only scores are external context, not a matched baseline or paired improvement claim.']
    (out/'comparison.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    q=commands.add_parser('prepare',help='Prepare pinned release or native JSONL/local images')
    q.add_argument('--revision',default='c6dd79d5e1c0cbb4e7ca234c9f53c418a75e30ce',help='Pinned audited HF revision; any override is resolved to an immutable SHA')
    q.add_argument('--source-jsonl',type=Path,help='Offline native records with local image paths')
    q.add_argument('--limit-pairs',type=int,help='Smoke only; outputs always marked preliminary')
    q.add_argument('--system-trust',action='store_true',help='Use OS certificate store without disabling TLS verification')
    q.add_argument('--out',type=Path,required=True)
    q=commands.add_parser('freeze');q.add_argument('--dataset',type=Path,required=True)
    q.add_argument('--spec',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q.add_argument('--allow-preliminary',action='store_true')
    for name in ('score-base','score-vlm','judge-inputs'):
        q=commands.add_parser(name)
        for flag in ('dataset','frozen','predictions','out'):q.add_argument('--'+flag,type=Path,required=True)
        q.add_argument('--run',required=True)
        if name=='score-base':
            q.add_argument('--official-predictions',type=Path)
            q.add_argument('--judged',type=Path);q.add_argument('--judge-metadata',type=Path)
        elif name=='score-vlm':
            for flag in ('base-predictions','decisions','judged','judge-metadata'):q.add_argument('--'+flag,type=Path,required=True)
        else:q.add_argument('--test',type=int,choices=[1,2],required=True)
    q=commands.add_parser('report');q.add_argument('metrics',nargs='+',type=Path)
    q.add_argument('--frozen',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.command=='prepare':prepare_command(a)
        elif a.command=='freeze':freeze(a.dataset,a.spec,a.out,a.allow_preliminary);print('Frozen four configurations.')
        elif a.command=='judge-inputs':judge_inputs(a)
        elif a.command=='score-base':
            r=scoring.score_base(a.dataset,a.frozen,a.run,a.predictions,a.out,
                official_path=a.official_predictions,judged_path=a.judged,metadata_path=a.judge_metadata)
            print(r['official_status'])
        elif a.command=='score-vlm':
            scoring.score_vlm(a.dataset,a.frozen,a.run,a.base_predictions,a.predictions,a.decisions,a.judged,a.judge_metadata,a.out)
            print('Official DiffSpot scores written; no benchmark thresholds selected.')
        elif a.command=='report':report(a.metrics,a.frozen,a.out)
    except (ValueError, KeyError, OSError, TypeError) as error:
        p.exit(1,f'DiffSpot command failed: {error}\n')


if __name__=='__main__':main()
