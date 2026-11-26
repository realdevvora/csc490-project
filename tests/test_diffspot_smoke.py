import importlib.util
import json
from pathlib import Path

import pytest
Image=pytest.importorskip('PIL.Image')
from regression_lens.data.diffspot import OPERATORS, prepare, read_jsonl, write_json, write_jsonl


@pytest.fixture
def workflow(tmp_path):
    spec=importlib.util.spec_from_file_location('smoke',Path(__file__).parents[1]/'scripts/diffspot_smoke.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for op in OPERATORS:
        for tier in ('easy','medium','hard'):
            for repeat in range(2):
                rows.append(dict(id=f'{op}-{tier}-{repeat}',difficulty=tier,task_type='visual_diff',
                    mutation_types=['mutate_'+op],mutation_dicts_json=[json.dumps(dict(type=op,property='unknown'))],
                    ground_truth_diff='Fixture',domain='fixture',target_bbox_x=0,target_bbox_y=0,target_bbox_w=2,target_bbox_h=2,
                    image_before=Image.new('RGB',(4,4)),image_after=Image.new('RGB',(4,4))))
    for i in range(8):
        row=dict(rows[0],id=f'control-{i}',difficulty='no_diff',task_type='no_diff',mutation_types=[],mutation_dicts_json=[])
        rows.append(row)
    data=tmp_path/'data';prepare(rows,data,'fixture',preliminary=True)
    return module,data,tmp_path


@pytest.mark.parametrize('n',[30,44,50])
def test_unique_reproducible_balanced_sample(workflow,n):
    module,data,root=workflow
    a=module.sample(data,root/'a',n);b=module.sample(data,root/'b',n)
    assert a['sample_ids']==b['sample_ids'] and len(set(a['sample_ids']))==n
    annotations={r['sample_id']:r for r in read_jsonl(data/'annotations.jsonl')}
    chosen=[annotations[x] for x in a['sample_ids']]
    assert sum(r['split']=='no_diff' for r in chosen)==5
    if n==44:
        assert len({(r['operator'],r['split']) for r in chosen if r['operator']})==39
    assert a['preliminary']
    assert all('operator' not in r and 'target_box' not in r for r in module.inputs_for(data,root/'a'))


def test_review_creates_portable_html_and_does_not_reselect_threshold(workflow):
    module,data,root=workflow;module.sample(data,root/'sample')
    inputs=module.inputs_for(data,root/'sample')
    preds=root/'predictions.jsonl';write_jsonl(preds,[dict(sample_id=r['sample_id'],score=.7,box=[0,0,2,2]) for r in inputs])
    val=root/'val.json';write_json(val,dict(n=1033,threshold=.8))
    module.review(data,root/'sample',preds,val,root/'review')
    metrics=json.loads((root/'review/metrics.json').read_text())
    assert metrics['preliminary'] and metrics['metrics']['threshold']==.8
    assert metrics['metrics']['detection']['recall']==0
    text=(root/'review/review.html').read_text(encoding='utf-8')
    assert 'data:image/png;base64,' in text and '<svg' in text


def test_sample_tampering_rejected(workflow):
    module,data,root=workflow;module.sample(data,root/'sample')
    path=root/'sample/inference.jsonl';rows=read_jsonl(path);rows[0]['reference_path']='bad.png';write_jsonl(path,rows)
    with pytest.raises(ValueError,match='differ'):module.inputs_for(data,root/'sample')


def test_framework_independent_adapter_loads_once_and_only_receives_images(workflow,monkeypatch):
    import sys
    from types import ModuleType
    module,data,root=workflow;module.sample(data,root/'sample')
    calls=[];loads=[]
    adapter=ModuleType('synthetic_owner_adapter')
    def factory(config):
        loads.append(config)
        def pair(expected,actual):
            assert isinstance(expected,Image.Image) and isinstance(actual,Image.Image)
            assert expected.mode==actual.mode=='RGB'
            calls.append(1)
            return dict(score=.25,box=None,op_pred=None)
        return pair
    adapter.factory=factory
    monkeypatch.setitem(sys.modules,'synthetic_owner_adapter',adapter)
    config=root/'config.json';write_json(config,dict(owner_setting='fixture'))
    module.predict(data,root/'sample','synthetic_owner_adapter:factory',config,root/'run')
    assert loads==[dict(owner_setting='fixture')] and len(calls)==44
    assert len(read_jsonl(root/'run/predictions.jsonl'))==44
    assert json.loads((root/'run/complete.json').read_text())['preliminary']


def test_notebook_cells_are_valid_python_and_have_no_executed_results():
    import ast
    path=Path(__file__).parents[1]/'notebooks/diffspot_smoke.ipynb'
    notebook=json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs']==[] and cell['execution_count'] is None
