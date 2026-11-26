"""DiffSpot contracts and offline scoring; never tune on benchmark annotations."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import numpy as np

from regression_lens.data.diffspot import (OPERATORS, TIERS, digest, fresh, load_prepared,
    read_json, read_jsonl, write_json, write_jsonl)
from regression_lens.evaluation.metrics import auroc, box_iou, detection_metrics, fpr_at_recall, macro_f1, point_in_box
from regression_lens.evaluation._diffspot_official.metrics import aggregate
from regression_lens.evaluation._diffspot_official.judge import reduce_judge_label

OFFICIAL_ROOT = Path(__file__).parent/'_diffspot_official'
OFFICIAL_FIELDS = {'id', 'split', 'model', 'prompt_version', 'raw_response', 'prediction'}
D2C_TYPES = ('background_color', 'border_radius', 'border_width', 'font_size', 'font_weight',
             'letter_spacing', 'line_height', 'margin', 'opacity', 'padding', 'position', 'text_color', 'width')
NO_DIFF_RESPONSE = 'No differences identified.'


def verify_official_sources():
    provenance = read_json(OFFICIAL_ROOT/'PROVENANCE.json')
    for source, expected in provenance['files'].items():
        relative = source.removeprefix('diffspot/')
        if digest(OFFICIAL_ROOT/relative) != expected:
            raise ValueError(f'Pinned upstream code/prompt was modified: {source}')
    return provenance


def index_rows(rows, key, expected):
    result = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or identifier in result:
            raise ValueError(f'Missing or duplicate {key}')
        result[identifier] = row
    if set(result) != set(expected):
        raise ValueError('Prediction/judge ID coverage differs from the entire prepared dataset')
    return result


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_geometry(row, annotation):
    width, height = [max(annotation['size_ref'][i], annotation['size_cand'][i]) for i in (0, 1)]
    box = row.get('box')
    if box is not None:
        if (not isinstance(box, list) or len(box) != 4 or not all(map(finite, box))
                or min(box[:2]) < 0 or min(box[2:]) <= 0
                or box[0]+box[2] > width or box[1]+box[3] > height):
            raise ValueError('Box must be positive and within original screenshot coordinates')
    point = row.get('point')
    if point is not None and (not isinstance(point, list) or len(point) != 2 or not all(map(finite, point))
                              or not (0 <= point[0] < width and 0 <= point[1] < height)):
        raise ValueError('Point is outside original screenshot coordinates')


def get_run(dataset, frozen_path, run_id):
    verify_official_sources()
    manifest, annotations = load_prepared(dataset)
    frozen = read_json(frozen_path)
    if frozen['dataset_manifest_sha256'] != digest(Path(dataset)/'manifest.json'):
        raise ValueError('Frozen benchmark references a different dataset manifest')
    if frozen['official_provenance_sha256'] != digest(OFFICIAL_ROOT/'PROVENANCE.json'):
        raise ValueError('Official scorer revision changed since freezing')
    if frozen['scorer_sha256'] != digest(Path(__file__)):
        raise ValueError('Project scorer changed since freezing; use the frozen checkout')
    runs = [r for r in frozen['runs'] if r['id'] == run_id]
    if len(runs) != 1:
        raise ValueError('Run is not registered uniquely in frozen benchmark')
    return manifest, annotations, frozen, runs[0]


def validate_base(predictions, annotations, run):
    indexed = index_rows(predictions, 'sample_id', [a['sample_id'] for a in annotations])
    for a in annotations:
        row = indexed[a['sample_id']]
        if not finite(row.get('score')):
            raise ValueError('Base score must be finite')
        if row.get('op_pred') is not None and row['op_pred'] not in D2C_TYPES:
            raise ValueError('Base op_pred must use the Design2Code taxonomy')
        if not run['supports_type'] and row.get('op_pred') is not None:
            raise ValueError('Run declares no type head but emits types')
        validate_geometry(row, a)
    return indexed


def localization(rows, annotations, changed):
    overlaps, hits, returned_iou, returned_hits, grounded = [], [], [], [], []
    for i, a in enumerate(annotations):
        if a['split'] == 'no_diff':
            continue
        row = rows[a['sample_id']]
        box, point = row.get('box'), row.get('point')
        if point is None and box is not None:
            point = [box[0]+box[2]/2, box[1]+box[3]/2]
        overlap = box_iou(box, a['target_box']) if box is not None else 0.
        hit = point_in_box(point, a['target_box']) if point is not None else False
        overlaps.append(overlap); hits.append(hit)
        if box is not None: returned_iou.append(overlap)
        if point is not None: returned_hits.append(hit)
        grounded.append(bool(changed[i]) and box is not None and overlap >= .5)
    return dict(target='Visible intersection of native target element, not changed-pixel bbox',
                n_changed=len(overlaps), n_box=len(returned_iou), n_point=len(returned_hits),
                mean_iou_all_changed=float(np.mean(overlaps)) if overlaps else None,
                pointing_all_changed=float(np.mean(hits)) if hits else None,
                mean_iou_conditional=float(np.mean(returned_iou)) if returned_iou else None,
                pointing_conditional=float(np.mean(returned_hits)) if returned_hits else None,
                localized_recall_iou50=float(np.mean(grounded)) if grounded else None)


def base_metrics(indexed, annotations, run):
    y = np.array([a['split'] != 'no_diff' for a in annotations])
    scores = np.array([indexed[a['sample_id']]['score'] for a in annotations])
    predicted = scores >= run['threshold']
    masks = {tier: np.array([a['split'] == tier for a in annotations]) for tier in TIERS}
    rates = {tier: float(predicted[mask].mean()) if mask.any() else None for tier, mask in masks.items()}
    rates['no_diff'] = 1-rates['no_diff'] if rates['no_diff'] is not None else None
    type_rows = [(a, indexed[a['sample_id']].get('op_pred')) for a in annotations if a['d2c_type']]
    types = None
    if run['supports_type'] and type_rows:
        # Missing types are wrong predictions, never removed from the denominator.
        names = sorted({a['d2c_type'] for a, _ in type_rows})
        types = macro_f1([names.index(a['d2c_type']) for a, _ in type_rows],
                         [names.index(p) if p in names else -1 for _, p in type_rows], names)
    slices = {}
    for name, mask in [('mapped', np.array([bool(a['d2c_type']) for a in annotations])),
                       ('unmapped', np.array([a['split'] != 'no_diff' and not a['d2c_type'] for a in annotations]))]:
        slices[name] = dict(n=int(mask.sum()), detection_recall=float(predicted[mask].mean()) if mask.any() else None)
    detector_report = aggregate([dict(split=a['split'], operator=a['operator'],
        is_true_positive=bool(predicted[i]) and bool(y[i]),
        is_true_negative=not bool(predicted[i]) and not bool(y[i])) for i,a in enumerate(annotations)]).to_dict()
    return dict(threshold=run['threshold'], threshold_source='frozen Design2Code validation',
                detector_only_report=detector_report,
                auroc_diff_vs_no_diff=auroc(y, scores), detection=detection_metrics(y, predicted),
                detection_by_tier=rates, detection_by_operator={op:dict(
                    n=sum(a['operator'] == op for a in annotations),
                    recall=float(np.mean([predicted[i] for i,a in enumerate(annotations) if a['operator'] == op]))
                    if any(a['operator'] == op for a in annotations) else None) for op in OPERATORS},
                no_diff_fpr_at_95_detection_recall=fpr_at_recall(scores[y], scores[~y]),
                localization=localization(indexed, annotations, predicted),
                type_classification=dict(status='available' if types is not None else 'unavailable',
                    mapped_support=len(type_rows), predicted_support=sum(p is not None for _,p in type_rows), metrics=types),
                taxonomy_slices=slices)


def unavailable_official(annotations):
    # Same report fields/denominators, but never disguise detection as mutation identification.
    report = aggregate([dict(split=a['split'], operator=a['operator'],
                             is_true_positive=False, is_true_negative=False) for a in annotations]).to_dict()
    for key in ('overall_accuracy', 'diff_overall_recall', 'no_diff_specificity', 'n_tp', 'n_tn'):
        report[key] = None
    for group in ('by_tier', 'by_operator'):
        for value in report[group].values():
            value['recall'] = value['n_tp'] = None
    return report


def validate_official(predictions, annotations, model, prompt_version):
    indexed = index_rows(predictions, 'id', [a['sample_id'] for a in annotations])
    for a in annotations:
        row = indexed[a['sample_id']]
        if set(row) != OFFICIAL_FIELDS:
            raise ValueError('Official predictions must contain exactly id/split/model/prompt_version/raw_response/prediction')
        if row['split'] != a['split'] or row['model'] != model or row['prompt_version'] != prompt_version:
            raise ValueError('Official split/model/prompt identity mismatch')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ('raw_response', 'prediction')):
            raise ValueError('Empty/failed responses cannot be scored as no-diff')
    return indexed


def judge_context(predictions_path, dataset, frozen_path, model, prompt_version):
    manifest, annotations = load_prepared(dataset)
    validate_official(read_jsonl(predictions_path), annotations, model, prompt_version)
    return dict(predictions_sha256=digest(predictions_path),
                dataset_manifest_sha256=digest(Path(dataset)/'manifest.json'),
                frozen_sha256=digest(frozen_path), official_provenance_sha256=digest(OFFICIAL_ROOT/'PROVENANCE.json'),
                model=model, prompt_version=prompt_version, judge_version='v2.0', judge_model='gpt-oss-120b',
                reasoning_effort='high', temperature=0, max_tokens=16384,
                judge_prompt_sha256=digest(OFFICIAL_ROOT/'prompts/judge.txt'))


def official_metrics(predictions_path, judged_path, metadata_path, dataset, frozen_path, annotations, model, prompt_version):
    expected_context = judge_context(predictions_path, dataset, frozen_path, model, prompt_version)
    if read_json(metadata_path) != expected_context:
        raise ValueError('Judge metadata does not match frozen source/predictions and official judge settings')
    judged = index_rows(read_jsonl(judged_path), 'id', [a['sample_id'] for a in annotations])
    reduced = []
    for a in annotations:
        item = judged[a['sample_id']]
        if item.get('split') != a['split'] or item.get('operator') != a['operator']:
            raise ValueError('Judge annotations/ID association mismatch')
        label = item.get('judge_label')
        if not isinstance(label, dict) or not isinstance(label.get('mutations'), list) or not isinstance(label.get('hallucinations'), list):
            raise ValueError('Missing/malformed official v2.0 judge_label')
        expected_mutations = 0 if a['split'] == 'no_diff' else len(a['gt_mutations'])
        if len(label['mutations']) != expected_mutations or any(
                m.get('verdict') not in ('correct', 'partial', 'missed') for m in label['mutations']):
            raise ValueError('Incomplete official mutation judgments')
        binary = reduce_judge_label(label, is_no_diff=a['split'] == 'no_diff')
        for key, value in binary.items():
            if key in item and item[key] != value:
                raise ValueError('Stored judge reduction disagrees with official reducer')
        reduced.append(dict(id=a['sample_id'], split=a['split'], operator=a['operator'], **binary))
    report = aggregate(reduced).to_dict()
    envelope = dict(judge_version='v2.0', judge_model='gpt-oss-120b', reasoning_effort='high',
                    n_predictions=len(annotations), n_judged=len(annotations), report=report)
    return envelope, reduced


def score_base(dataset, frozen_path, run_id, predictions_path, out, *, official_path=None, judged_path=None, metadata_path=None):
    manifest, annotations, frozen, run = get_run(dataset, frozen_path, run_id)
    predictions = read_jsonl(predictions_path)
    indexed = validate_base(predictions, annotations, run)
    metrics = base_metrics(indexed, annotations, run)
    official = unavailable_official(annotations)
    status = 'unavailable: no free-form mutation-identification output/judgments supplied'
    envelope = None
    if any(p is not None for p in (official_path, judged_path, metadata_path)):
        if not all(p is not None for p in (official_path, judged_path, metadata_path)):
            raise ValueError('Supply official predictions, judged JSONL and judge metadata together')
        model = run['id']
        validate_official(read_jsonl(official_path), annotations, model, run['base_prompt_version'])
        envelope, _ = official_metrics(official_path, judged_path, metadata_path, dataset, frozen_path,
                                       annotations, model, run['base_prompt_version'])
        official = envelope['report']; status = 'official description scoring'
    result = dict(test=1, run=run_id, preliminary=manifest['preliminary'], official_status=status,
                  official_metrics=official, supplementary=metrics,
                  predictions_sha256=digest(predictions_path), frozen_sha256=digest(frozen_path))
    result['artifact_hashes'] = {key:digest(path) if path else None for key,path in
        [('official_predictions',official_path),('judged',judged_path),('judge_metadata',metadata_path)]}
    out = fresh(out)
    write_json(out/'metrics.json', result)
    if envelope: write_json(out/'official_scores.json', envelope)
    return result


def score_vlm(dataset, frozen_path, run_id, base_path, predictions_path, decisions_path, judged_path, metadata_path, out):
    manifest, annotations, frozen, run = get_run(dataset, frozen_path, run_id)
    base = validate_base(read_jsonl(base_path), annotations, run)
    model = run_id+'+'+frozen['kimi']['model']
    prompt_version = frozen['kimi']['prompt_version']
    predictions = validate_official(read_jsonl(predictions_path), annotations, model, prompt_version)
    decisions = index_rows(read_jsonl(decisions_path), 'sample_id', [a['sample_id'] for a in annotations])
    final, accepted, counts = {}, [], Counter()
    base_hash = digest(base_path)
    for a in annotations:
        sid = a['sample_id']; item = decisions[sid]; decision = item.get('decision')
        if decision not in ('accept', 'reject', 'unsure'):
            raise ValueError('Verification decision must be accept/reject/unsure; errors are incomplete runs')
        if item.get('source_prediction_sha256') != base_hash:
            raise ValueError('Verification is not bound to the supplied base predictions')
        keep = decision == 'accept'
        if keep and (base[sid].get('box') is None or item.get('box') != base[sid]['box']):
            raise ValueError('Kimi must preserve a real detector-proposed box')
        if not keep and item.get('box') is not None:
            raise ValueError('Rejected/unsure issues must not retain a final box')
        if item.get('description') != predictions[sid]['prediction']:
            raise ValueError('Verifier description differs from official prediction text')
        if not keep and item['description'] != NO_DIFF_RESPONSE:
            raise ValueError('Non-accepted proposals must use the frozen no-difference response')
        counts[decision] += 1; accepted.append(keep)
        final[sid] = dict(sample_id=sid, box=base[sid]['box'] if keep else None,
                          point=base[sid].get('point') if keep else None)
    envelope, reduced = official_metrics(predictions_path, judged_path, metadata_path, dataset, frozen_path,
                                         annotations, model, prompt_version)
    before = base_metrics(base, annotations, run)
    grounded = [r['is_true_positive'] and accepted[i] and final[a['sample_id']]['box'] is not None
                and box_iou(final[a['sample_id']]['box'], a['target_box']) >= .5
                for i,(a,r) in enumerate(zip(annotations,reduced)) if a['split'] != 'no_diff']
    changed = np.array([a['split'] != 'no_diff' for a in annotations])
    result = dict(test=2, run=run_id, preliminary=manifest['preliminary'],
        official_status='official description scoring', official_metrics=envelope['report'],
        supplementary=dict(verification_counts=dict(counts), abstention_rate=counts['unsure']/len(annotations),
            accepted_issue_detection=detection_metrics(changed,np.array(accepted)),
            proposal_localization=before['localization'],
            final_localization=localization(final,annotations,accepted),
            correctly_described_and_localized_recall_iou50=float(np.mean(grounded)) if grounded else None,
            coordinate_refinement=False),
        base_predictions_sha256=digest(base_path), predictions_sha256=digest(predictions_path),
        frozen_sha256=digest(frozen_path))
    result['artifact_hashes'] = dict(decisions=digest(decisions_path), judged=digest(judged_path),
                                    judge_metadata=digest(metadata_path))
    out = fresh(out)
    write_json(out/'metrics.json', result)
    write_json(out/'official_scores.json', envelope)
    # Export only the six official fields, never our verification sidecar.
    write_jsonl(out/'official_predictions.jsonl', [predictions[a['sample_id']] for a in annotations])
    return result
