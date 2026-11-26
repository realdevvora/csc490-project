"""Native DiffSpot preparation, separate from Design2Code's metadata contract."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

OPERATORS = ('font_weight', 'font_size', 'letter_spacing', 'line_height', 'text',
             'color', 'opacity', 'gradient', 'position', 'spacing', 'justify', 'border', 'rounded')
TIERS = ('easy', 'medium', 'hard', 'no_diff')
TAXONOMY_PATH = Path(__file__).with_name('diffspot_taxonomy_v1.json')
TAXONOMY = json.loads(TAXONOMY_PATH.read_text(encoding='utf-8'))
MAPPING_VERSION = TAXONOMY['version']
EXACT = TAXONOMY['exact']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def fresh(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Use a new or empty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def mapped_type(operator, mutation):
    """Strict property-aware mapping; no semantic guessing or GT at inference."""
    prop = str(mutation.get('property', '')).lower().replace('_', '-')
    if operator in EXACT:
        return EXACT[operator], 'same visual property'
    mapped = TAXONOMY['property_rules'].get(operator, {}).get(prop)
    if mapped:
        return mapped, 'explicit supported CSS property'
    return None, 'no exact Design2Code class for this operator/property'


def normalize(row):
    """Inspect annotations for scoring only; do not pass this record to models."""
    sample_id = row.get('id')
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError('DiffSpot id must be a nonempty string')
    # Match the official loader, including older revisions with 28 "unknown"
    # controls: no_diff task_type overrides difficulty routing.
    tier = 'no_diff' if row['task_type'] == 'no_diff' else row['difficulty']
    if tier not in TIERS:
        raise ValueError(f'Invalid difficulty: {tier}')
    no_diff = row['task_type'] == 'no_diff'
    if row['task_type'] not in ('no_diff', 'visual_diff') or no_diff != (tier == 'no_diff'):
        raise ValueError('task_type/difficulty mismatch')
    mutations = [json.loads(x) if isinstance(x, str) else x for x in (row.get('mutation_dicts_json') or [])]
    types = row.get('mutation_types') or []
    if no_diff:
        if mutations or types:
            raise ValueError('No-diff record contains mutations')
        operator, box, mapped, reason = None, None, None, 'no-diff control'
    else:
        if len(mutations) != 1 or len(types) != 1:
            raise ValueError('Pinned release expects one mutation per changed pair')
        operator = types[0].removeprefix('mutate_')
        if operator not in OPERATORS or mutations[0].get('type') != operator:
            raise ValueError('Unknown or inconsistent native mutation type')
        box = [row['target_bbox_'+x] for x in ('x', 'y', 'w', 'h')]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box) or min(box[2:]) <= 0:
            raise ValueError('Invalid target-element box')
        mapped, reason = mapped_type(operator, mutations[0])
    return dict(sample_id=sample_id, split=tier, task_type=row['task_type'],
                operator=operator, source_target_box=box, target_box=box, d2c_type=mapped, mapping_reason=reason,
                gt_description=row['ground_truth_diff'], gt_mutations=mutations,
                source_mutation_types=list(types), source_difficulty=row['difficulty'], domain=row.get('domain') or '')


def prepare(rows, out, source_revision, *, preliminary=False, source_name='tencent/DiffSpot'):
    out = fresh(out)
    (out/'images').mkdir()
    annotations, inputs, ids = [], [], set()
    for row in rows:
        annotation = normalize(row)
        sample_id = annotation['sample_id']
        if sample_id in ids:
            raise ValueError('Duplicate dataset id')
        ids.add(sample_id)
        # Original UUIDs are retained for official compatibility; image names
        # are opaque hashes and never include operators or browser names.
        opaque = hashlib.sha256(sample_id.encode()).hexdigest()
        images = [row['image_before'].convert('RGB'), row['image_after'].convert('RGB')]
        paths = [f'images/{opaque}_{side}.png' for side in ('a', 'b')]
        for image, path in zip(images, paths):
            image.save(out/path)
        width, height = [max(image.size[i] for image in images) for i in (0, 1)]
        box = annotation['target_box']
        if box:
            x0,y0=max(0,box[0]),max(0,box[1])
            x1,y1=min(width,box[0]+box[2]),min(height,box[1]+box[3])
            if x1 <= x0 or y1 <= y0:
                raise ValueError('Target element has no visible screenshot intersection; inspect the release before benchmarking')
            annotation['target_box']=[x0,y0,x1-x0,y1-y0]
        annotation['target_clipped_to_canvas'] = annotation['target_box'] != annotation['source_target_box']
        annotation.update(size_ref=list(images[0].size), size_cand=list(images[1].size))
        annotations.append(annotation)
        inputs.append(dict(sample_id=sample_id, reference_path=paths[0], candidate_path=paths[1],
                           size_ref=list(images[0].size), size_cand=list(images[1].size),
                           reference_sha256=digest(out/paths[0]), candidate_sha256=digest(out/paths[1])))
    counts = Counter(a['split'] for a in annotations)
    op_counts = Counter(a['operator'] for a in annotations if a['operator'])
    if not preliminary and (dict(counts) != dict(easy=1300, medium=1300, hard=1300, no_diff=500)
                            or op_counts != Counter({x: 300 for x in OPERATORS})):
        raise ValueError('Release counts differ from canonical 4,400-pair benchmark')
    if not annotations:
        raise ValueError('Empty dataset')
    write_jsonl(out/'inference.jsonl', inputs)
    write_jsonl(out/'annotations.jsonl', annotations)
    coverage = Counter((a['operator'], a['d2c_type'], a['mapping_reason']) for a in annotations if a['operator'])
    write_json(out/'mapping-coverage.json', [dict(operator=k[0], d2c_type=k[1], reason=k[2], n=v)
                                           for k, v in sorted(coverage.items(), key=lambda item:str(item[0]))])
    manifest = dict(format_version=1, dataset=source_name, source_revision=source_revision,
                    preliminary=preliminary, n=len(inputs), counts=dict(counts),
                    operator_counts=dict(op_counts), mapping_version=MAPPING_VERSION,
                    mapping_sha256=digest(TAXONOMY_PATH),
                    localization_target='visible intersection of native target element with union screenshot canvas',
                    inference_sha256=digest(out/'inference.jsonl'),
                    annotations_sha256=digest(out/'annotations.jsonl'))
    write_json(out/'manifest.json', manifest)
    return manifest


def load_prepared(path):
    path = Path(path)
    manifest = read_json(path/'manifest.json')
    for filename, key in [('annotations.jsonl', 'annotations_sha256'), ('inference.jsonl', 'inference_sha256')]:
        if digest(path/filename) != manifest[key]:
            raise ValueError(f'Prepared dataset changed: {filename}')
    annotations = read_jsonl(path/'annotations.jsonl')
    inputs = read_jsonl(path/'inference.jsonl')
    expected = {r['sample_id'] for r in annotations}
    if len(expected) != manifest['n'] or len(inputs) != manifest['n'] or {r['sample_id'] for r in inputs} != expected:
        raise ValueError('Prepared manifest coverage mismatch')
    counts = Counter(a['split'] for a in annotations)
    if dict(counts) != manifest['counts']:
        raise ValueError('Prepared tier counts mismatch')
    if not manifest['preliminary'] and (dict(counts) != dict(easy=1300, medium=1300, hard=1300, no_diff=500)
            or Counter(a['operator'] for a in annotations if a['operator']) != Counter({x:300 for x in OPERATORS})):
        raise ValueError('A full benchmark must contain the canonical 4,400 records')
    return manifest, annotations
