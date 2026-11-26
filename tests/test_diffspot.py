"""Offline DiffSpot contracts; fixtures never represent measured model performance."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

Image = pytest.importorskip('PIL.Image')

from regression_lens.data.diffspot import (OPERATORS, mapped_type, normalize, prepare,
    digest, load_prepared, read_json, read_jsonl, write_json, write_jsonl)
from regression_lens.evaluation import diffspot as ds
from regression_lens.evaluation._diffspot_official.metrics import aggregate
from regression_lens.evaluation._diffspot_official.judge import reduce_judge_label
from regression_lens.evaluation._diffspot_official.data import _row_to_item


@pytest.fixture
def cli():
    spec=importlib.util.spec_from_file_location('diffspot_cli',Path(__file__).parents[1]/'scripts/diffspot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def native(identifier, operator='font_size', tier='easy', prop='font-size'):
    control = tier == 'no_diff'
    return dict(id=identifier, difficulty=tier, task_type='no_diff' if control else 'visual_diff',
        mutation_types=[] if control else ['mutate_'+operator],
        mutation_dicts_json=[] if control else [json.dumps(dict(type=operator, property=prop))],
        ground_truth_diff='' if control else 'Element looks different.', domain='test',
        target_bbox_x=-1 if control else 2, target_bbox_y=-1 if control else 1,
        target_bbox_w=-1 if control else 3, target_bbox_h=-1 if control else 2,
        image_before=Image.new('RGB',(8,6),'white'), image_after=Image.new('RGB',(8,6),'white'))


@pytest.mark.parametrize('op,prop,target', [
    ('rounded','border-radius','border_radius'), ('font_size','font-size','font_size'),
    ('color','color','text_color'),('color','background-color','background_color'),
    ('color','unknown',None),('spacing','padding-left','padding'),('spacing','gap',None),
    ('spacing','margin','margin'),('spacing','horizontal padding','padding'),
    ('spacing','bottom margin','margin'),('color','text-shade','text_color'),
    ('color','background-shade','background_color'),('spacing','gap between items',None),
    ('border','border-color',None),('border','border-width',None),
    ('gradient','background-image',None),('text','text-content',None),('justify','justify-content',None),
])
def test_taxonomy_never_guesses_unrelated_properties(op,prop,target):
    assert mapped_type(op,{'property':prop})[0] == target


def test_unknown_difficulty_controls_match_official_loader():
    row=native('control',tier='no_diff');row['difficulty']='unknown'
    assert normalize(row)['split'] == _row_to_item(row).split == 'no_diff'
    for op in OPERATORS:
        row=native(op,operator=op)
        assert normalize(row)['operator'] == _row_to_item(row).operator


@pytest.fixture
def prepared(tmp_path,cli):
    rows=[native('a'),native('b',operator='gradient',tier='medium',prop='background-image'),
          native('c',operator='rounded',tier='hard',prop='border-radius'),native('d',tier='no_diff')]
    data=tmp_path/'data';prepare(rows,data,'fixture-sha',preliminary=True)
    metrics=tmp_path/'validation.json';write_json(metrics,dict(n=1033,threshold=.5))
    prompt=tmp_path/'prompt.txt';prompt.write_text('Fixed synthetic prompt.')
    spec=tmp_path/'spec.json'
    write_json(spec,dict(selection_dataset='design2code_val',
        runs=[dict(id=f'model{i}',family=f'family{i}',checkpoint_sha256=str(i)*64,
                  seed_policy='fixed synthetic seed',preprocessing='native',box_rule='fixture',
                  validation_metrics_file='validation.json',supports_type=(i==1)) for i in range(4)],
        kimi=dict(model='kimi-k2.5',prompt_version='fixture-v1',prompt_file='prompt.txt')))
    frozen=tmp_path/'frozen.json';cli.freeze(data,spec,frozen,True)
    predictions=[dict(sample_id=x,score=.9 if x!='d' else .1,box=[2.,1.,3.,2.],point=[3.,2.]) for x in 'abcd']
    base=tmp_path/'base.jsonl';write_jsonl(base,predictions)
    return dict(root=tmp_path,data=data,frozen=frozen,base=base,spec=spec)


def test_inference_manifest_has_no_target_or_label_leak(prepared):
    for row in read_jsonl(prepared['data']/'inference.jsonl'):
        assert set(row)=={'sample_id','reference_path','candidate_path','size_ref','size_cand','reference_sha256','candidate_sha256'}
        assert not any(op in row['reference_path'] for op in OPERATORS)


def test_full_release_check_rejects_smoke_and_invalid_target(tmp_path):
    with pytest.raises(ValueError,match='4,400'):
        prepare([native('a')],tmp_path/'wrong','fixture')
    row=native('bad');row['target_bbox_w']=-1
    with pytest.raises(ValueError,match='Invalid target'):
        prepare([row],tmp_path/'bad','fixture',preliminary=True)


def test_partly_offscreen_target_is_preserved_and_clipped_only_for_localization(tmp_path):
    row=native('offscreen');row['target_bbox_y']=4;row['target_bbox_h']=5
    prepare([row],tmp_path/'data','fixture',preliminary=True)
    _,annotations=load_prepared(tmp_path/'data')
    assert annotations[0]['source_target_box']==[2,4,3,5]
    assert annotations[0]['target_box']==[2,4,3,2]
    assert annotations[0]['target_clipped_to_canvas']
    assert annotations[0]['gt_mutations']==normalize(row)['gt_mutations']


def test_freeze_rejects_smoke_without_explicit_flag_and_never_overwrites(prepared,cli):
    with pytest.raises(ValueError,match='Smoke'):
        cli.freeze(prepared['data'],prepared['spec'],prepared['root']/'badfreeze.json')
    with pytest.raises(FileExistsError):
        cli.freeze(prepared['data'],prepared['spec'],prepared['frozen'],True)
    spec=read_json(prepared['spec']);spec['selection_dataset']='diffspot'
    write_json(prepared['root']/'invalid.json',spec)
    with pytest.raises(ValueError,match='Selection'):
        cli.freeze(prepared['data'],prepared['root']/'invalid.json',prepared['root']/'bad.json',True)


def test_base_keeps_unmapped_examples_and_official_scores_unavailable(prepared):
    result=ds.score_base(prepared['data'],prepared['frozen'],'model0',prepared['base'],prepared['root']/'score1')
    assert result['official_metrics']['overall_accuracy'] is None
    assert result['official_metrics']['n_total']==4
    assert result['official_metrics']['by_operator']['gradient']['n_pairs']==1
    assert result['supplementary']['detector_only_report']['overall_accuracy']==1
    assert result['supplementary']['localization']['n_box']==3
    assert result['supplementary']['taxonomy_slices']['unmapped']['n']==1
    assert result['supplementary']['type_classification']['status']=='unavailable'


def judged_fixture(prepared):
    root=prepared['root']
    _,annotations=load_prepared(prepared['data'])
    predictions=[];decisions=[];judged=[]
    # correct / partial / missed plus a hallucination on the no-diff control.
    verdicts=['correct','partial','missed',None]
    for a,verdict in zip(annotations,verdicts):
        sid=a['sample_id']
        predictions.append(dict(id=sid,split=a['split'],model='model0+kimi-k2.5',prompt_version='fixture-v1',
                                raw_response='Synthetic raw response',prediction='A visible change.'))
        decisions.append(dict(sample_id=sid,decision='accept',box=[2.,1.,3.,2.],description='A visible change.',
                              source_prediction_sha256=digest(prepared['base'])))
        label=dict(mutations=[] if verdict is None else [dict(type=a['operator'],gt='fixture',verdict=verdict,vlm_match='fixture')],
                   hallucinations=['imaginary difference'] if verdict is None else [],
                   summary=dict(correct=int(verdict=='correct'),partial=int(verdict=='partial'),
                                missed=int(verdict=='missed'),hallucinated=int(verdict is None)))
        judged.append(dict(id=sid,split=a['split'],operator=a['operator'],judge_label=label,
                           **reduce_judge_label(label,is_no_diff=a['split']=='no_diff')))
    official=root/'official.jsonl';sidecar=root/'decisions.jsonl';labels=root/'judged.jsonl';metadata=root/'judge-meta.json'
    write_jsonl(official,predictions);write_jsonl(sidecar,decisions);write_jsonl(labels,judged)
    write_json(metadata,ds.judge_context(official,prepared['data'],prepared['frozen'],'model0+kimi-k2.5','fixture-v1'))
    return dict(official=official,sidecar=sidecar,labels=labels,metadata=metadata)


def score_fixture(prepared,files,out='score2'):
    return ds.score_vlm(prepared['data'],prepared['frozen'],'model0',prepared['base'],files['official'],
        files['sidecar'],files['labels'],files['metadata'],prepared['root']/out)


def test_vlm_reports_exact_official_fields_and_reduction(prepared):
    files=judged_fixture(prepared);result=score_fixture(prepared,files)
    expected=aggregate(read_jsonl(files['labels'])).to_dict()
    assert result['official_metrics']==expected
    assert expected['overall_accuracy']==.25
    assert expected['diff_overall_recall']==pytest.approx(1/3)
    assert expected['no_diff_specificity']==0
    assert expected['by_tier']['easy']['recall']==1
    assert expected['by_tier']['medium']['recall']==0
    assert expected['by_tier']['hard']['recall']==0
    exported=read_jsonl(prepared['root']/'score2/official_predictions.jsonl')
    assert all(set(r)==ds.OFFICIAL_FIELDS for r in exported)
    envelope=read_json(prepared['root']/'score2/official_scores.json')
    assert set(envelope)=={'judge_version','judge_model','reasoning_effort','n_predictions','n_judged','report'}


@pytest.mark.parametrize('problem',['missing_judge','duplicate_judge','changed_box','wrong_split','wrong_metadata','empty_response','stored_wrong_reduction'])
def test_incomplete_or_mismatched_benchmark_artifacts_fail(prepared,problem):
    files=judged_fixture(prepared)
    if problem in ('missing_judge','duplicate_judge','stored_wrong_reduction'):
        rows=read_jsonl(files['labels'])
        if problem=='missing_judge':rows.pop()
        elif problem=='duplicate_judge':rows[-1]=rows[0]
        else:rows[0]['is_true_positive']=False
        write_jsonl(files['labels'],rows)
    elif problem=='changed_box':
        rows=read_jsonl(files['sidecar']);rows[0]['box']=[0.,0.,2.,2.];write_jsonl(files['sidecar'],rows)
    elif problem=='wrong_metadata':
        meta=read_json(files['metadata']);meta['judge_model']='kimi-k2.5';write_json(files['metadata'],meta)
    else:
        rows=read_jsonl(files['official'])
        if problem=='wrong_split':rows[0]['split']='no_diff'
        else:rows[0]['raw_response']=''
        write_jsonl(files['official'],rows)
    with pytest.raises(ValueError):score_fixture(prepared,files)


def test_rejection_preserves_proposal_iou_but_reduces_grounded_recall(prepared):
    files=judged_fixture(prepared)
    official=read_jsonl(files['official']);decisions=read_jsonl(files['sidecar'])
    decisions[0].update(decision='unsure',box=None,description=ds.NO_DIFF_RESPONSE)
    official[0]['prediction']=ds.NO_DIFF_RESPONSE
    write_jsonl(files['sidecar'],decisions);write_jsonl(files['official'],official)
    write_json(files['metadata'],ds.judge_context(files['official'],prepared['data'],prepared['frozen'],'model0+kimi-k2.5','fixture-v1'))
    result=score_fixture(prepared,files)
    assert result['supplementary']['proposal_localization']['mean_iou_all_changed']==1
    assert result['supplementary']['final_localization']['mean_iou_all_changed']==pytest.approx(2/3)
    assert result['supplementary']['abstention_rate']==.25


def test_missing_type_predictions_are_errors_not_filtered_out(prepared):
    rows=read_jsonl(prepared['base'])
    rows[0]['op_pred']='font_size'
    write_jsonl(prepared['base'],rows)
    result=ds.score_base(prepared['data'],prepared['frozen'],'model1',prepared['base'],prepared['root']/'typed')
    types=result['supplementary']['type_classification']
    assert types['mapped_support']==2 and types['predicted_support']==1
    assert types['metrics']['macro_f1']==.5


def test_report_pairs_tests_and_marks_incomplete_comparison(prepared,cli):
    ds.score_base(prepared['data'],prepared['frozen'],'model0',prepared['base'],prepared['root']/'score1')
    score_fixture(prepared,judged_fixture(prepared))
    cli.report([prepared['root']/'score1',prepared['root']/'score2'],prepared['frozen'],prepared['root']/'report')
    result=read_json(prepared['root']/'report/comparison.json')
    assert not result['complete_four_pipeline_comparison']
    assert result['paired'][0]['official_deltas']['overall_accuracy'] is None


def test_vendored_sources_are_byte_exact_and_prompts_have_identical_body():
    ds.verify_official_sources()
    def body(name):
        return (ds.OFFICIAL_ROOT/'prompts'/name).read_text(encoding='utf-8').split('>>> PROMPT BEGINS BELOW THIS LINE >>>')[1].strip()
    assert body('vlm_diff.txt')==body('vlm_nodiff.txt')


def test_ground_truth_cannot_silently_change_after_preparation(prepared):
    path=prepared['data']/'annotations.jsonl';rows=read_jsonl(path);rows[0]['target_box']=[0,0,1,1];write_jsonl(path,rows)
    with pytest.raises(ValueError,match='changed'):load_prepared(prepared['data'])
