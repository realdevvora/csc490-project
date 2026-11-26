import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from regression_lens.evaluation.dino_boxes import heatmap_box


def test_largest_component_masks_padding_and_clips_odd_image_edge():
    values = np.array([[1., 0., 99.], [.8, 0., 99.], [0., 0., .7]])
    valid = np.array([[True, True, False], [True, True, False], [True, True, True]])
    box, audit = heatmap_box(values, valid, width=5, height=5)
    assert box == [0., 0., 2., 4.]
    assert audit['components'] == 2 and audit['map_threshold'] == .5
    values[:] = 0; values[2, 2] = 1
    assert heatmap_box(values, valid, 5, 5)[0] == [4., 4., 1., 1.]


def test_zero_map_fallback_uses_valid_cell_not_padding():
    box, audit = heatmap_box(np.zeros((1, 2)), np.array([[False, True]]), 3, 1)
    assert box == [2., 0., 1., 1.]
    assert audit['zero_map_fallback']


def test_components_are_four_connected_and_ties_use_row_major_order():
    box, _ = heatmap_box(np.eye(2), np.ones((2, 2), bool), 4, 4)
    assert box == [0., 0., 2., 2.]


@pytest.mark.parametrize('values,valid', [
    (np.ones((1, 1)), np.ones((1, 1), bool)),  # wrong shape
    (np.ones((2, 2)), np.zeros((2, 2), bool)),
    (np.full((2, 2), np.nan), np.ones((2, 2), bool)),
    (-np.ones((2, 2)), np.ones((2, 2), bool)),
])
def test_bad_maps_are_rejected(values, valid):
    with pytest.raises(ValueError):
        heatmap_box(values, valid, 4, 4)


def exporter():
    path = Path(__file__).resolve().parents[1]/'scripts/export_dino_boxes.py'
    spec = importlib.util.spec_from_file_location('export_dino_boxes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved_baseline(tmp_path):
    evaluation, data = tmp_path/'dinov2_val', tmp_path/'data'
    (evaluation/'maps').mkdir(parents=True); data.mkdir()
    records = [dict(sample_id=label, parent_id='page', label=label,
                    operator='width' if label == 'regression' else None,
                    size_ref=[5, 3], size_cand=[5, 3], diff_bbox=[2, 0, 2, 2],
                    bbox_after=[2, 0, 2, 2], reflow=False, family='layout', magnitude='small')
               for label in ['regression', 'benign', 'no_change']]
    metadata = data/'master_dataset_with_benign.json'
    metadata.write_text(json.dumps(records))
    split = tmp_path/'split.json'; split.write_text(json.dumps(dict(val=['page'])))
    config = dict(sample_ids=[r['sample_id'] for r in records], output_scale=2, subset='val',
                  split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                  metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
    (evaluation/'config.json').write_text(json.dumps(config))
    (evaluation/'complete.json').write_text('{}')
    predictions = [dict(sample_id=r['sample_id'], score=s, point=point)
                   for r, s, point in zip(records, [.9, .2, 0.], [[3., 1.], None, None])]
    (evaluation/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    for i in range(3):
        np.savez_compressed(evaluation/'maps'/f'{i:05d}.npz',
                            cosine=np.array([[0., 1., 0.], [0., 0., 0.]]), valid=np.ones((2, 3), bool))
    return evaluation, data, split, tmp_path/'boxes'


def test_export_rescores_iou_without_mutating_source_predictions(saved_baseline):
    evaluation, data, split, out = saved_baseline
    before = (evaluation/'predictions.jsonl').read_bytes()
    exporter().export([evaluation], data, split, out)
    metrics = json.loads((out/'dinov2_val__A/metrics.json').read_text())
    assert metrics['localization']['mean_iou'] == 1.
    assert metrics['localization']['pointing_accuracy'] == 1.
    assert metrics['localization']['n_box'] == 1
    assert metrics['detection']['f1'] == 1.
    assert (evaluation/'predictions.jsonl').read_bytes() == before
    with pytest.raises(ValueError, match='Output must be empty'):
        exporter().export([evaluation], data, split, out)


def test_missing_maps_fail_before_output_creation(saved_baseline):
    evaluation, data, split, out = saved_baseline
    (evaluation/'maps/00001.npz').unlink()
    with pytest.raises(ValueError, match='missing maps'):
        exporter().export([evaluation], data, split, out)
    assert not out.exists()


def test_config_order_not_prediction_order_controls_map_identity(saved_baseline):
    evaluation, data, split, out = saved_baseline
    path = evaluation/'predictions.jsonl'
    path.write_text('\n'.join(reversed(path.read_text().splitlines()))+'\n')
    exporter().export([evaluation], data, split, out)
    rows = [json.loads(r) for r in (out/'dinov2_val__A/predictions.jsonl').read_text().splitlines()]
    assert rows[-1]['sample_id'] == 'regression' and rows[-1]['box'] == [2., 0., 2., 2.]
