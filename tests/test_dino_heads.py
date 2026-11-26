import importlib.util
from pathlib import Path
import random
import json
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip('torch')
from regression_lens.models.dino_head import SpatialHead, paired_features, weak_targets, balanced_location_loss, head_loss


def runner():
    spec = importlib.util.spec_from_file_location('dino_heads', Path(__file__).parents[1]/'scripts/dino_heads.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(label='regression'):
    return dict(sample_id=label, parent_id='page', size_ref=[500, 500], size_cand=[500, 500],
                label=label, reflow=False, bbox_after=[16, 16, 16, 16], diff_bbox=[16, 16, 16, 16])


def test_features_symmetric_and_invariant_to_positive_scale():
    x, y = torch.randn(2, 4, 3, 3), torch.randn(2, 4, 3, 3)
    torch.testing.assert_close(paired_features(x, y), paired_features(y, x))
    torch.testing.assert_close(paired_features(x, y), paired_features(x*3, y*2))
    assert paired_features(x, y).shape == (2, 9, 3, 3)


def test_weak_targets_and_reflow_ignore():
    r = record()
    t = weak_targets(r, [0, 0, 448, 448], 16)
    assert t['changed'] == 1 and t['target'].sum() == 1
    r.update(reflow=True, diff_bbox=[0, 0, 400, 400])
    t = weak_targets(r, [0, 0, 448, 448], 16)
    assert t['mask'][0, 0] == 0 and t['mask'][1, 1] == 1
    assert t['mask'][26, 26] == 1
    t = weak_targets(record('benign'), [0, 0, 448, 448], 16)
    assert t['changed'] == 0 and t['target'].sum() == 0


def test_partial_padding_union_and_outside_window():
    r = record('no_change')
    r.update(size_ref=[8, 8], size_cand=[16, 16])
    t = weak_targets(r, [0, 0, 448, 448], 16)
    assert t['valid'].sum() == 1
    t = weak_targets(record(), [100, 100, 548, 548], 16)
    assert t['changed'] == 0 and t['target'].sum() == 0


def test_ignored_pixels_have_zero_gradient():
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    target = torch.tensor([[[1., 0.], [0., 0.]]])
    mask = torch.tensor([[[1., 0.], [1., 0.]]])
    balanced_location_loss(logits, target, mask).backward()
    assert logits.grad[0, 0, 1] == logits.grad[0, 1, 1] == 0
    assert logits.grad[0, 0, 0] < 0 and logits.grad[0, 1, 0] > 0


def test_training_manifest_stable_train_only():
    m = runner()
    rows = [record(label) for label in ('regression', 'benign', 'no_change')]
    other = dict(record(), sample_id='outside', parent_id='validation')
    split = dict(train=['page'], val=['validation'])
    a = m.training_manifest(rows+[other], split, 'split', 'metadata')
    b = m.training_manifest(list(reversed(rows+[other])), split, 'split', 'metadata')
    assert a == b and len(a['windows']) == 6
    assert {w['parent_id'] for w in a['windows']} == {'page'}
    assert a['windows'][1]['crop'] == a['windows'][3]['crop']


def test_ablation_batches_match_regression_stream_and_budget():
    m = runner()
    pools = dict(positive=list(range(20)), benign=list(range(20, 30)), no_change=list(range(30, 40)))
    b = m.sample_batch(pools, 'B', random.Random(490), random.Random(999))
    c = m.sample_batch(pools, 'C', random.Random(490), random.Random(999))
    assert len(b) == len(c) == 32 and b[:16] == c[:16]
    assert sum(20 <= i < 30 for i in b) == 8
    assert not any(20 <= i < 30 for i in c)


@pytest.mark.parametrize('ref,cand,box', [
    ([1280, 801], [1280, 809], [0, 801, 1280, 8]),
    ([1280, 1570], [1288, 1570], [1280, 0, 8, 1570]),
    ([1280, 809], [1280, 801], [0, 801, 1280, 8]),
])
def test_anchor_covers_added_or_removed_candidate_border(ref, cand, box):
    m = runner()
    r = record()
    r.update(size_ref=ref, size_cand=cand, diff_bbox=box, bbox_after=box)
    manifest = m.training_manifest([r], dict(train=['page'], val=[]), 'split', 'metadata')
    crop = next(w['crop'] for w in manifest['windows'] if w['kind'] == 'anchor')
    assert m.crop_coverage(box, crop) > 0
    for patch in (14, 16):
        target = weak_targets(r, crop, patch)
        assert target['changed'] == 1 and target['target'].sum() > 0


def test_invalid_off_image_anchor_fails_before_extraction():
    m = runner()
    r = record()
    r.update(diff_bbox=[1000, 1000, 10, 10], bbox_after=[1000, 1000, 10, 10])
    with pytest.raises(ValueError, match='misses target'):
        m.training_manifest([r], dict(train=['page'], val=[]), 'split', 'metadata')


def test_covered_reference_crop_is_preserved_despite_larger_candidate():
    m = runner()
    r = record()
    r.update(size_cand=[1000, 1000], diff_bbox=[450, 450, 10, 10])
    manifest = m.training_manifest([r], dict(train=['page'], val=[]), 's', 'm')
    anchor = next(w for w in manifest['windows'] if w['kind'] == 'anchor')
    assert anchor['crop'] == [52, 52, 500, 500]


def test_controls_keep_exact_geometry_of_their_anchor():
    m = runner()
    r = record()
    r.update(size_cand=[508, 500], diff_bbox=[500, 0, 8, 500])
    control = record('benign')
    control['size_cand'] = [1400, 1500]
    manifest = m.training_manifest([r, control], dict(train=['page'], val=[]), 's', 'm')
    crops = [w['crop'] for w in manifest['windows'] if w['kind'] == 'anchor']
    assert crops[0] == crops[1] and m.crop_coverage(r['diff_bbox'], crops[0]) > 0


def test_tiny_synthetic_overfit_and_frozen_inputs():
    torch.set_num_threads(2)
    torch.manual_seed(490)
    head = SpatialHead(channels=4)
    # Emulate detached outputs of a frozen backbone.
    backbone = torch.nn.Conv2d(3, 4, 1).requires_grad_(False)
    x = backbone(torch.randn(4, 3, 4, 4)).detach()
    y = x.clone()
    y[2:] = -x[2:]
    batch = dict(changed=torch.tensor([0., 0., 1., 1.]),
                 target=torch.tensor([0., 0., 1., 1.])[:, None, None].expand(4, 4, 4),
                 mask=torch.ones(4, 4, 4))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    head.eval()
    first = head_loss(head(x, y), batch)[0].item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = head_loss(head(x, y), batch)[0]
        loss.backward()
        optimizer.step()
    assert loss.item() < first*.2
    assert all(p.grad is None for p in backbone.parameters())
    head.eval()
    expected = head(x, y)[0]
    clone = SpatialHead(channels=4).eval()
    clone.load_state_dict(head.state_dict())
    torch.testing.assert_close(expected, clone(x, y)[0])


def test_checkpoint_selection_across_seeds_and_earlier_step_tie(tmp_path):
    m = runner()
    checkpoints = {}
    for condition in ('B', 'C'):
        for step in (500, 1000, 2000):
            for seed in (490, 491, 492):
                name = f'{condition}_{seed}_{step}'
                checkpoints[name] = dict(config=dict(condition=condition, seed=seed, smoke=False), step=step)
                directory = tmp_path/name
                directory.mkdir()
                value = .8 if condition == 'C' or step != 500 else .6
                (directory/'metrics.json').write_text(json.dumps(dict(
                    auroc=dict(regression_vs_benign=value), benign_fpr_at_95_recall=dict(fpr=.2))))
    (tmp_path/'config.json').write_text(json.dumps(dict(smoke=False, checkpoints=checkpoints)))
    (tmp_path/'complete.json').write_text(json.dumps(dict(pairs=1033)))
    m.report_command(SimpleNamespace(evaluation=tmp_path))
    result = json.loads((tmp_path/'selection.json').read_text())
    assert result['selected']['B']['step'] == 1000
    assert result['selected']['C']['step'] == 500
    assert result['selected']['C']['auroc_std'] == pytest.approx(0)


def test_smoke_cannot_be_reported_as_final(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(smoke=True)))
    (tmp_path/'complete.json').write_text('{}')
    with pytest.raises(ValueError, match='Smoke'):
        runner().report_command(SimpleNamespace(evaluation=tmp_path))
