"""Guided 30-50 pair DiffSpot smoke: sample, predict, review.

Operational checks only. Never choose weights, thresholds or prompts on these
results; changes informed by them must be declared exploratory under ADR 0007.
"""
import argparse
import base64
import csv
import html
from pathlib import Path
import random

from regression_lens.data.diffspot import (OPERATORS, digest, fresh, load_prepared,
    read_json, read_jsonl, write_json, write_jsonl)
from regression_lens.evaluation.diffspot import base_metrics, validate_base


def sample(dataset, out, n=44, seed=490):
    if not 30 <= n <= 50:
        raise ValueError('Choose 30-50 pairs')
    manifest, annotations = load_prepared(dataset)
    rng = random.Random(seed)
    buckets = {}
    for op in OPERATORS:
        for tier in ('easy', 'medium', 'hard'):
            rows = sorted((a['sample_id'] for a in annotations if a['operator']==op and a['split']==tier))
            rng.shuffle(rows); buckets[(op,tier)] = rows
    controls = sorted(a['sample_id'] for a in annotations if a['split']=='no_diff')
    rng.shuffle(controls)
    count_controls = 5
    if len(controls) < count_controls or any(not rows for rows in buckets.values()):
        raise ValueError('Sampling requires full operator/tier coverage and five controls')
    # Alternate operators and cycle tiers; 44 selects all 39 cells once.
    order = [(op,('easy','medium','hard')[(t+i)%3]) for t in range(3) for i,op in enumerate(OPERATORS)]
    chosen = []
    while len(chosen) < n-count_controls:
        progressed = False
        for key in order:
            if buckets[key] and len(chosen) < n-count_controls:
                chosen.append(buckets[key].pop()); progressed = True
        if not progressed:
            raise ValueError('Not enough unique changed pairs')
    chosen += controls[:count_controls]
    rng.shuffle(chosen)
    out = fresh(out)
    inputs = {r['sample_id']:r for r in read_jsonl(Path(dataset)/'inference.jsonl')}
    write_jsonl(out/'inference.jsonl',[inputs[x] for x in chosen])
    value = dict(preliminary=True, dataset_manifest_sha256=digest(Path(dataset)/'manifest.json'),
        source_revision=manifest['source_revision'], seed=seed, sample_ids=chosen,
        n=n, no_diff=count_controls, method='round-robin native operator/tier cells; fixed seed',
        warning='Smoke only; no DiffSpot-based model/threshold/prompt selection')
    write_json(out/'sample.json',value)
    print(f'Saved {n} pairs ({n-count_controls} changed + 5 controls) to {out}')
    return value


def inputs_for(dataset, sample_dir):
    sample_dir = Path(sample_dir)
    selection = read_json(sample_dir/'sample.json')
    if selection['dataset_manifest_sha256'] != digest(Path(dataset)/'manifest.json'):
        raise ValueError('Sample belongs to a different prepared dataset')
    rows = read_jsonl(sample_dir/'inference.jsonl')
    originals = {r['sample_id']:r for r in read_jsonl(Path(dataset)/'inference.jsonl')}
    if [r['sample_id'] for r in rows] != selection['sample_ids'] or len(rows)!=len(set(selection['sample_ids'])):
        raise ValueError('Sample IDs changed')
    if any(r != originals.get(r['sample_id']) for r in rows):
        raise ValueError('Inference inputs differ from the prepared source')
    return rows


def predict(dataset, sample_dir, adapter, model_config, out):
    """Load any framework through an explicit owner-supplied factory.

    Factory(config_dict) -> callable(expected_PIL_RGB, actual_PIL_RGB) -> dict.
    Only images enter prediction; sample IDs/annotations never enter the model.
    """
    import importlib
    from PIL import Image
    rows = inputs_for(dataset, sample_dir)
    module, separator, factory = adapter.partition(':')
    if not separator or not module or not factory:
        raise ValueError('Adapter must be an importable module:factory')
    config = read_json(model_config)
    predictor = getattr(importlib.import_module(module), factory)(config)
    if not callable(predictor):
        raise ValueError('Adapter factory must return a callable predictor')
    out = fresh(out)
    write_json(out/'run.json', dict(preliminary=True, adapter=adapter,
        model_config_sha256=digest(model_config),
        sample_sha256=digest(Path(sample_dir)/'sample.json'),
        runner_sha256=digest(Path(__file__))))
    predictions=[]
    for index,row in enumerate(rows):
        images=[]
        for side in ('reference','candidate'):
            path=Path(dataset)/row[side+'_path']
            if digest(path)!=row[side+'_sha256']:
                raise ValueError('Input image changed')
            with Image.open(path) as image:images.append(image.convert('RGB'))
        result=predictor(images[0], images[1])
        if not isinstance(result,dict) or not set(result)<= {'score','box','point','op_pred'}:
            raise ValueError('Predictor returns score and optional box/point/op_pred only')
        # Geometry validation uses only image dimensions, never labels.
        from regression_lens.evaluation.diffspot import finite,validate_geometry,D2C_TYPES
        if not finite(result.get('score')):raise ValueError('Prediction score must be finite')
        if result.get('op_pred') is not None and result['op_pred'] not in D2C_TYPES:
            raise ValueError('Base op_pred uses the shared Design2Code taxonomy')
        validate_geometry(result,dict(size_ref=images[0].size,size_cand=images[1].size))
        predictions.append(dict(sample_id=row['sample_id'],**result))
        print(f'{index+1}/{len(rows)} score={result["score"]:.4f}',flush=True)
    write_jsonl(out/'predictions.jsonl',predictions)
    write_json(out/'complete.json',dict(preliminary=True,n=len(predictions)))


def review(dataset,sample_dir,predictions,validation_metrics,out,supports_type=False,from_full=False):
    inputs=inputs_for(dataset,sample_dir)
    _,all_annotations=load_prepared(dataset)
    lookup={a['sample_id']:a for a in all_annotations}
    annotations=[lookup[r['sample_id']] for r in inputs]
    rows=read_jsonl(predictions)
    if from_full:
        if len({r['sample_id'] for r in rows}) != len(rows) or {r['sample_id'] for r in rows} != set(lookup):
            raise ValueError('--from-full requires exactly the full dataset predictions')
        wanted={a['sample_id'] for a in annotations};rows=[r for r in rows if r['sample_id'] in wanted]
    validation=read_json(validation_metrics)
    if validation.get('n') != 1033:
        raise ValueError('Use Design2Code validation metrics, not DiffSpot metrics')
    run=dict(threshold=validation['threshold'],supports_type=supports_type)
    from regression_lens.evaluation.diffspot import finite
    if not finite(run['threshold']):raise ValueError('Validation threshold must be finite')
    indexed=validate_base(rows,annotations,run)
    metrics=base_metrics(indexed,annotations,run)
    out=fresh(out)
    write_json(out/'metrics.json',dict(preliminary=True,n=len(rows),metrics=metrics,
        sample_sha256=digest(Path(sample_dir)/'sample.json'),predictions_sha256=digest(predictions),
        validation_metrics_sha256=digest(validation_metrics),
        official_description_scores='Unavailable: no descriptive output/official judge labels; use full DiffSpot workflow for Test 2'))
    from regression_lens.evaluation.metrics import box_iou
    cases=[];cards=[]
    for row,a in zip(inputs,annotations):
        p=indexed[a['sample_id']];box=p.get('box');gt=a['target_box']
        overlap=box_iou(box,gt) if box is not None and gt is not None else None
        cases.append(dict(sample_id=a['sample_id'],tier=a['split'],operator=a['operator'],
                          score=p['score'],predicted_changed=p['score']>=run['threshold'],iou=overlap))
        views=[]
        for side,title in [('reference','Expected'),('candidate','Actual')]:
            w,h=row['size_ref' if side=='reference' else 'size_cand']
            raw=(Path(dataset)/row[side+'_path']).read_bytes()
            # Self-contained HTML can be copied from lab to local for review.
            markup=f'<svg viewBox="0 0 {w} {h}"><image width="{w}" height="{h}" href="data:image/png;base64,{base64.b64encode(raw).decode()}"/>'
            for region,color in [(gt,'#15803d'),(box,'#2563eb')]:
                if region is not None:
                    x,y,bw,bh=region
                    markup+=f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="none" stroke="{color}" stroke-width="3"/>'
            views.append(f'<div><b>{title}</b>{markup}</svg></div>')
        label=html.escape(f"{a['sample_id']} | {a['split']} | {a['operator']} | score={p['score']:.4f} | IoU={overlap}")
        cards.append(f'<section><h3>{label}</h3><p>{html.escape(a["gt_description"] or "No-diff control")}</p><div class="pair">'+''.join(views)+'</div></section>')
    with (out/'cases.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Preliminary DiffSpot smoke</title>'
        '<style>body{font-family:system-ui;margin:24px}section{border-top:1px solid #ccc;padding:16px 0}.pair{display:flex;gap:12px}.pair div{width:50%}svg{width:100%}h3{overflow-wrap:anywhere}</style>'
        '<h1>Preliminary smoke review — not model-selection data</h1><p>Green: visible target element. Blue: predicted box. '
        'Scores use the unchanged Design2Code threshold. Do not tune from this review.</p>'+''.join(cards),encoding='utf-8')
    print(f"Smoke F1={metrics['detection']['f1']:.4f}; AUROC={metrics['auroc_diff_vs_no_diff']}; review: {out/'review.html'}")


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    for name in ('sample','predict','review'):
        q=commands.add_parser(name);q.add_argument('--dataset',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
        if name=='sample':q.add_argument('--n',type=int,default=44);q.add_argument('--seed',type=int,default=490)
        else:q.add_argument('--sample',type=Path,required=True)
        if name=='predict':
            q.add_argument('--adapter',required=True,help='Importable module:factory returning a pair predictor')
            q.add_argument('--model-config',type=Path,required=True)
        if name=='review':
            q.add_argument('--predictions',type=Path,required=True);q.add_argument('--validation-metrics',type=Path,required=True)
            q.add_argument('--supports-type',action='store_true');q.add_argument('--from-full',action='store_true')
    a=p.parse_args()
    try:
        if a.command=='sample':sample(a.dataset,a.out,a.n,a.seed)
        elif a.command=='predict':predict(a.dataset,a.sample,a.adapter,a.model_config,a.out)
        else:review(a.dataset,a.sample,a.predictions,a.validation_metrics,a.out,a.supports_type,a.from_full)
    except (ValueError,KeyError,OSError) as error:p.exit(1,f'Smoke check failed: {error}\n')


if __name__=='__main__':main()
