"""Frozen DINO experiments: manifest, extract, train, evaluate, report.

Run --help or docs/results/dino/head-runs.md. DiffSpot is deliberately excluded.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import Dataset

from regression_lens.data.records import load_records, operator_names
from regression_lens.data.splits import TEAM_SPLIT, load_split
from regression_lens.evaluation.scoring import score_predictions, validate_predictions
from regression_lens.evaluation.metrics import point_in_box, auroc
from regression_lens.evaluation.tiling import window_origins, add_patch_map, summarize_map
from regression_lens.models.dino_head import SpatialHead, weak_targets, head_loss
from regression_lens.models.dino_runtime import (Backbone, MODELS, REPO, sha, read, write,
    empty_output, provenance, crop_image, open_pair)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def training_manifest(records, split, split_hash, metadata_hash, limit=None):
    pages = sorted(split['train'])[:limit]
    grouped = defaultdict(list)
    for r in sorted(records, key=lambda r: r['sample_id']):
        if r['parent_id'] in pages:
            grouped[r['parent_id']].append(r)
    windows = []
    for page, rows in sorted(grouped.items()):
        regressions = [r for r in rows if r['label'] == 'regression']
        for record in rows:
            # Stable per-page randomness keeps the uniform window matched across labels.
            rng = random.Random(f'490:{page}')
            w, h = record['size_ref']
            uniform = [rng.randint(0, max(0, w-448)), rng.randint(0, max(0, h-448))]
            if not regressions:
                raise ValueError(f'No anchor mutations for {page}')
            anchor = record if record['label'] == 'regression' else regressions[
                int(hashlib.sha256(record['sample_id'].encode()).hexdigest(), 16) % len(regressions)]
            box = anchor['bbox_after'] if anchor['reflow'] else anchor['diff_bbox']
            # Candidate-only strips are real image area too. Clamping to the
            # reference alone can exclude exactly the changed border region.
            centered = [max(0, min(math.floor(box[i]+box[i+2]/2-224), max(0, anchor['size_ref'][i]-448))) for i in (0, 1)]
            c = [*centered, centered[0]+448, centered[1]+448]
            # Preserve already covered frozen crops; only repair zero coverage.
            # Controls use their chosen anchor's exact geometry, not their own
            # browser-dependent extent, to keep paired locations matched.
            if crop_coverage(box, c) == 0 or crop_coverage(anchor['diff_bbox'], c) == 0:
                extent = [max(anchor['size_ref'][i], anchor['size_cand'][i]) for i in (0, 1)]
                centered = [max(0, min(math.floor(box[i]+box[i+2]/2-224), max(0, extent[i]-448))) for i in (0, 1)]
            if record['label'] == 'regression':
                c = [*centered, centered[0]+448, centered[1]+448]
                if crop_coverage(box, c) == 0 or crop_coverage(record['diff_bbox'], c) == 0:
                    raise ValueError(f"Anchor crop misses target/changed box: {record['sample_id']}")
            for kind, origin in (('uniform', uniform), ('anchor', centered)):
                windows.append(dict(sample_id=record['sample_id'], parent_id=page,
                    label=record['label'], kind=kind, anchor_id=anchor['sample_id'],
                    crop=[*origin, origin[0]+448, origin[1]+448]))
    return dict(version=2, crop_policy='reference-clamped anchor; union-extent fallback for zero coverage',
                seed=490, split_sha256=split_hash, metadata_sha256=metadata_hash,
                pages=pages, smoke=limit is not None, windows=windows)


def crop_coverage(box, crop):
    x, y, w, h = box
    intersection = max(0, min(x+w, crop[2])-max(x, crop[0]))*max(0, min(y+h, crop[3])-max(y, crop[1]))
    return intersection/(w*h) if w*h > 0 else 0.


def dataset(a):
    records = load_records(a.data_root)
    split = load_split(REPO/TEAM_SPLIT, records)
    return records, split


def manifest_command(a):
    records, split = dataset(a)
    value = training_manifest(records, split, sha(REPO/TEAM_SPLIT),
        sha(a.data_root/'master_dataset_with_benign.json'), a.limit_pages)
    if a.manifest.exists():
        if read(a.manifest) != value:
            raise ValueError('Existing window manifest differs; preserve it')
    else:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        write(a.manifest, value)
    print(f"Frozen {len(value['windows'])} windows on {len(value['pages'])} training pages")


def extract_command(a):
    records, split = dataset(a)
    manifest = read(a.manifest)
    expected = training_manifest(records, split, sha(REPO/TEAM_SPLIT),
        sha(a.data_root/'master_dataset_with_benign.json'), len(manifest['pages']) if manifest['smoke'] else None)
    if manifest != expected:
        raise ValueError('Window manifest incompatible with dataset or committed split')
    patch = 14 if a.model == 'dinov2' else 16
    estimate = len(manifest['windows'])*(2*768*(448//patch)**2*4+8192)
    cap = 96*1024**3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(a.out.parent).free
    print(f'Cache estimate {estimate/1024**3:.2f} GiB; free {free/1024**3:.2f} GiB', flush=True)
    if estimate > cap or free < estimate*1.2:
        raise RuntimeError('Cache exceeds 96 GiB cap or requires more free space (20% headroom)')
    out = empty_output(a.out)
    backbone = Backbone(a.model, a.device)
    config = dict(model=a.model, model_revision=MODELS[a.model][1], manifest_sha256=sha(a.manifest),
                  manifest=manifest, backbone=backbone.config(), provenance=provenance())
    write(out/'config.json', config)
    by_id = {r['sample_id']: r for r in records}
    began = perf_counter()
    training_index = []
    for index, item in enumerate(manifest['windows']):
        record = by_id[item['sample_id']]
        images = open_pair(a.data_root, record)
        features = backbone.features([crop_image(im, item['crop']) for im in images]).cpu()
        assert not features.requires_grad and not any(p.requires_grad for p in backbone.model.parameters())
        payload = dict(reference=features[0], candidate=features[1], **weak_targets(record, item['crop'], backbone.patch))
        if not torch.isfinite(features).all():
            raise ValueError(f'Nonfinite features: {item}')
        torch.save(payload, out/f'{index:05d}.pt')
        training_index.append(dict(label=item['label'], changed=bool(payload['changed'].item())))
        if (index+1) % 50 == 0:
            print(f"Extracted {index+1}/{len(manifest['windows'])}", flush=True)
    write(out/'training_index.json', training_index)
    write(out/'complete.json', dict(windows=len(manifest['windows']), seconds=perf_counter()-began))


class CachedWindows(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.config = read(self.root/'config.json')
        complete = read(self.root/'complete.json')
        self.items = self.config['manifest']['windows']
        self.training_index = read(self.root/'training_index.json')
        if complete['windows'] != len(self.items):
            raise ValueError('Incomplete cache')
        if len(self.training_index) != len(self.items):
            raise ValueError('Incomplete cache index')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return torch.load(self.root/f'{index:05d}.pt', map_location='cpu', weights_only=True)


def sampling_pools(cache):
    pools = defaultdict(list)
    for i, item in enumerate(cache.training_index):
        # Window status was computed from geometry during extraction.
        pools['positive' if item['changed'] else item['label']].append(i)
    return pools


def sample_batch(pools, condition, positive_rng, negative_rng):
    positive = positive_rng.choices(pools['positive'], k=16)
    if condition == 'B':
        negative = negative_rng.choices(pools['benign'], k=8)+negative_rng.choices(pools['no_change'], k=8)
    else:
        negative = negative_rng.choices(pools['no_change'], k=16)
    return positive+negative


def load_batch(cache, indices, device):
    rows = [cache[i] for i in indices]
    return {key: torch.stack([r[key] for r in rows]).to(device) for key in rows[0]}


def train_command(a):
    cache = CachedWindows(a.cache)
    if cache.config['manifest']['split_sha256'] != sha(REPO/TEAM_SPLIT):
        raise ValueError('Cache uses a different split')
    pools = sampling_pools(cache)
    required = ['positive', 'no_change']+(['benign'] if a.condition == 'B' else [])
    if any(not pools[k] for k in required):
        raise ValueError(f'Missing training groups: {required}')
    seed_all(a.seed)
    device = torch.device(a.device)
    out = empty_output(a.out)
    model = SpatialHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    config = dict(condition=a.condition, seed=a.seed, steps=a.steps, batch_size=32,
        cache_config_sha256=sha(a.cache/'config.json'), cache=cache.config,
        optimizer='AdamW', lr=.0003, weight_decay=.01, location_weight=1,
        architecture='paired1537-conv128-conv64-dropout0.1-location+pooled_detection',
        provenance=provenance(), smoke=cache.config['manifest']['smoke'] or a.steps != 2000)
    write(out/'config.json', config)
    positive_rng, negative_rng = random.Random(a.seed), random.Random(a.seed+10000)
    # Fixed training diagnostic windows are identical across B/C and do not update weights.
    common = [i for i, item in enumerate(cache.items) if item['label'] != 'benign']
    diagnostic_indices = random.Random(490).sample(common, min(32, len(common)))
    diagnostic = load_batch(cache, diagnostic_indices, device)
    checkpoints = sorted({s for s in (500, 1000, 2000, a.steps) if s <= a.steps})
    began = perf_counter()
    if a.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with (out/'learning.jsonl').open('w') as log:
        for step in range(1, a.steps+1):
            model.train()
            batch = load_batch(cache, sample_batch(pools, a.condition, positive_rng, negative_rng), device)
            optimizer.zero_grad(set_to_none=True)
            loss, det, loc = head_loss(model(batch['reference'], batch['candidate'], batch['valid']), batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            if not torch.isfinite(grad):
                raise RuntimeError('Nonfinite gradients')
            optimizer.step()
            if step == 1 or step % 25 == 0 or step in checkpoints:
                model.eval()
                with torch.no_grad():
                    outputs = model(diagnostic['reference'], diagnostic['candidate'], diagnostic['valid'])
                    diagnostic_loss = head_loss(outputs, diagnostic)[0].item()
                row = dict(step=step, loss=loss.item(), detection=det.item(), localization=loc.item(),
                           gradient_norm=grad.item(), fixed_training_loss=diagnostic_loss,
                           fixed_training_auroc=auroc(diagnostic['changed'].cpu().numpy().astype(bool),
                                                     outputs[0].sigmoid().cpu().numpy()))
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(row, flush=True)
            if step in checkpoints:
                torch.save(dict(state_dict=model.state_dict(), step=step, config=config), out/f'step_{step}.pt')
    write(out/'complete.json', dict(steps=a.steps, seconds=perf_counter()-began,
        checkpoints=checkpoints, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None))


def evaluate_command(a):
    records, split = dataset(a)
    by_id = {r['sample_id']: r for r in records}
    pages = sorted(split['val'])[:a.limit_pages]
    selected = sorted([r for r in records if r['parent_id'] in pages], key=lambda r: r['sample_id'])
    paths = sorted(a.runs.glob('*/seed*/step_*.pt'))
    if not paths:
        raise ValueError('No checkpoints: expected RUNS/B/seed490/step_500.pt etc.')
    heads, configs = {}, {}
    manifest_hashes = set()
    device = torch.device(a.device)
    for path in paths:
        finished = read(path.parent/'complete.json')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        config = payload['config']
        manifest_hashes.add(config['cache']['manifest_sha256'])
        if payload['step'] not in finished['checkpoints']:
            raise ValueError('Checkpoint is not part of a completed run')
        if config != read(path.parent/'config.json'):
            raise ValueError('Checkpoint/run configuration mismatch')
        if config['cache']['model'] != a.model or config['cache']['model_revision'] != MODELS[a.model][1]:
            raise ValueError('Backbone/checkpoint mismatch')
        manifest = config['cache']['manifest']
        if manifest['split_sha256'] != sha(REPO/TEAM_SPLIT) or manifest['metadata_sha256'] != sha(a.data_root/'master_dataset_with_benign.json'):
            raise ValueError('Checkpoint dataset/split mismatch')
        name = f"{config['condition']}_seed{config['seed']}_step{payload['step']}"
        if name in heads:
            raise ValueError(f'Duplicate checkpoint {name}')
        heads[name] = SpatialHead().to(device).eval()
        heads[name].load_state_dict(payload['state_dict'])
        configs[name] = dict(config=config, step=payload['step'], checkpoint=str(path), checkpoint_sha256=sha(path))
    if len(manifest_hashes) != 1:
        raise ValueError('All evaluated heads must use the same frozen training windows')
    out = empty_output(a.out)
    backbone = Backbone(a.model, a.device)
    for entry in configs.values():
        if entry['config']['cache']['backbone'] != backbone.config():
            raise ValueError('Preprocessing/backbone contract changed')
    write(out/'config.json', dict(model=a.model, checkpoints=configs, pages=pages,
        sample_ids=[r['sample_id'] for r in selected], smoke=a.limit_pages is not None,
        split_sha256=sha(REPO/TEAM_SPLIT), provenance=provenance()))
    predictions = {name: [] for name in heads}
    began = perf_counter()
    if a.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for pair_index, record in enumerate(selected):
            images = open_pair(a.data_root, record)
            width, height = max(im.width for im in images), max(im.height for im in images)
            shape = (math.ceil(height/2), math.ceil(width/2))
            sums = {name: np.zeros(shape, np.float32) for name in heads}
            counts = np.zeros(shape, np.uint16)
            valid = np.zeros(shape, bool)
            for image in images:
                valid[:math.ceil(image.height/2), :math.ceil(image.width/2)] = True
            scores = {name: 0. for name in heads}
            for x, y in window_origins(width, height):
                crop = [x, y, x+448, y+448]
                features = backbone.features([crop_image(im, crop) for im in images])
                # Validity uses dimensions only; never annotations or labels.
                synthetic = dict(size_ref=images[0].size, size_cand=images[1].size, label='no_change')
                weights = weak_targets(synthetic, crop, backbone.patch)['valid'][None].to(device)
                old_counts = counts.copy()
                for name, head in heads.items():
                    detection, location = head(features[0:1], features[1:2], weights)
                    scores[name] = max(scores[name], detection.sigmoid().item())
                    counts[:] = old_counts
                    add_patch_map(sums[name], counts, location.sigmoid()[0].cpu().numpy(), x, y, backbone.patch)
            assert np.all(counts[valid] > 0)
            for name in heads:
                values = sums[name]/np.maximum(counts, 1)
                point = summarize_map(values, valid)['point']
                point = [min(point[0], width-.5), min(point[1], height-.5)]
                predictions[name].append(dict(sample_id=record['sample_id'], score=scores[name], point=point))
                # Persist maps for all checkpoints for review and exact provenance.
                directory = out/name
                directory.mkdir(exist_ok=True)
                np.savez_compressed(directory/f'{pair_index:05d}.npz', location=values, valid=valid)
            print(f'{pair_index+1}/{len(selected)} {record["sample_id"]}', flush=True)
    for name, rows in predictions.items():
        validate_predictions(rows, by_id, operator_names(records), expected_ids=[r['sample_id'] for r in selected])
        with (out/name/'predictions.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
        metrics = score_predictions(rows, records, operator_names(records), expected_ids=[r['sample_id'] for r in selected])
        write(out/name/'metrics.json', metrics)
        breakdown = defaultdict(list)
        for prediction in rows:
            record = by_id[prediction['sample_id']]
            if record['label'] != 'regression':
                continue
            box = record['bbox_after'] if record['reflow'] else record['diff_bbox']
            area = record['diff_bbox'][2]*record['diff_bbox'][3]/(record['size_ref'][0]*record['size_ref'][1])
            size_band = 'small_le_0.1pct' if area <= .001 else 'medium_le_1pct' if area <= .01 else 'large_gt_1pct'
            for key in (f"operator/{record['operator']}", f"change_size/{size_band}"):
                breakdown[key].append((prediction['score'] >= metrics['threshold'], point_in_box(prediction['point'], box)))
        write(out/name/'breakdowns.json', {key: dict(n=len(values), recall=float(np.mean([v[0] for v in values])),
            pointing_accuracy=float(np.mean([v[1] for v in values]))) for key, values in breakdown.items()})
    write(out/'complete.json', dict(pairs=len(selected), heads=len(heads), seconds=perf_counter()-began,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None))


def report_command(a):
    config, complete = read(a.evaluation/'config.json'), read(a.evaluation/'complete.json')
    if config['smoke']:
        raise ValueError('Smoke evaluations cannot select final checkpoints')
    groups = defaultdict(list)
    for name, entry in config['checkpoints'].items():
        run = entry['config']
        if run['smoke']:
            raise ValueError('Smoke-trained checkpoints cannot be final results')
        metrics = read(a.evaluation/name/'metrics.json')
        groups[(run['condition'], entry['step'])].append((run['seed'], name, metrics))
    summary = []
    for (condition, step), values in sorted(groups.items()):
        if sorted(v[0] for v in values) != [490, 491, 492]:
            raise ValueError(f'Missing/duplicate seeds for {condition}/{step}')
        aurocs = [v[2]['auroc']['regression_vs_benign'] for v in values]
        fprs = [v[2]['benign_fpr_at_95_recall']['fpr'] for v in values]
        def flatten(value, prefix=''):
            result = {}
            for key, item in value.items():
                name = f'{prefix}/{key}' if prefix else key
                if isinstance(item, dict):
                    result.update(flatten(item, name))
                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                    result[name] = item
            return result
        flat = [flatten(v[2]) for v in values]
        common = set.intersection(*(set(v) for v in flat))
        aggregates = {key: dict(mean=float(np.mean([v[key] for v in flat])),
                               std=float(np.std([v[key] for v in flat], ddof=1))) for key in sorted(common)}
        summary.append(dict(condition=condition, step=step, auroc_mean=float(np.mean(aurocs)),
            auroc_std=float(np.std(aurocs, ddof=1)), benign_fpr95_mean=float(np.mean(fprs)),
            aggregate_metrics=aggregates,
            seeds={str(seed): dict(name=name, metrics=metrics) for seed, name, metrics in values}))
    selected = {}
    for condition in ('B', 'C'):
        choices = [r for r in summary if r['condition'] == condition]
        if sorted(r['step'] for r in choices) != [500, 1000, 2000]:
            raise ValueError(f'Missing planned checkpoint steps for {condition}')
        selected[condition] = sorted(choices, key=lambda r: (-r['auroc_mean'], r['benign_fpr95_mean'], r['step']))[0]
    write(a.evaluation/'selection.json', dict(checkpoint_candidates=summary, selected=selected,
        interpretation='validation-selected results; no independent internal test', complete=complete))
    lines = ['| Condition | Step | AUROC mean ± sample SD | Benign FPR@95R mean |', '|---|---:|---:|---:|']
    for condition, row in selected.items():
        lines.append(f"| {condition} | {row['step']} | {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f} | {row['benign_fpr95_mean']:.4f} |")
    (a.evaluation/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system-trust', action='store_true')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('manifest', 'extract', 'train', 'evaluate', 'report'):
        q = sub.add_parser(name)
        if name in ('manifest', 'extract', 'evaluate'):
            q.add_argument('--data-root', type=Path, required=True)
        if name in ('manifest', 'evaluate'):
            q.add_argument('--limit-pages', type=int)
        if name in ('manifest', 'extract'):
            q.add_argument('--manifest', type=Path, required=True)
        if name in ('extract', 'evaluate'):
            q.add_argument('--model', choices=MODELS, required=True)
        if name in ('extract', 'train', 'evaluate'):
            q.add_argument('--out', type=Path, required=True)
            q.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        if name == 'train':
            q.add_argument('--cache', type=Path, required=True)
            q.add_argument('--condition', choices=('B', 'C'), required=True)
            q.add_argument('--seed', type=int, choices=(490, 491, 492), default=490)
            q.add_argument('--steps', type=int, default=2000)
        if name == 'evaluate':
            q.add_argument('--runs', type=Path, required=True)
        if name == 'report':
            q.add_argument('--evaluation', type=Path, required=True)
    a = p.parse_args()
    if getattr(a, 'limit_pages', None) is not None and a.limit_pages <= 0:
        p.error('--limit-pages must be positive')
    if getattr(a, 'steps', 1) <= 0:
        p.error('--steps must be positive')
    if a.system_trust:
        import truststore
        truststore.inject_into_ssl()
    # Needed by deterministic CUDA linear backward; no effect on current jobs.
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    globals()[a.command+'_command'](a)


if __name__ == '__main__':
    main()
