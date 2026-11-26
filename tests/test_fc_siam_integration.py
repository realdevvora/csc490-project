"""Compatibility checks for the imported FC-Siam workflow; no trained weights."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_script(path):
    spec = importlib.util.spec_from_file_location('fc_siam_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exporter():
    return load_script(REPO / 'scripts/export_fc_siam_predictions.py')


def test_export_uses_definitions_without_training_or_automatic_export(exporter):
    notebook = json.loads(exporter.NOTEBOOK.read_text(encoding='utf-8'))
    cells = list(exporter.export_cells(notebook))
    assert cells[0][0] == 'config'
    sources = '\n'.join(source for _, source in cells)
    assert 'optimizer.step()' not in sources
    assert 'run_dir.mkdir' not in sources
    assert "random.sample(train_rows, 30)" not in sources
    prediction = [source for role, source in cells if role == 'predict']
    assert len(prediction) == 1
    tree = ast.parse(prediction[0])
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == 'write_predictions'
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), str(exporter.NOTEBOOK), 'exec')


def test_export_rejects_unclassified_notebook_code(exporter):
    notebook = {'cells': [{'cell_type': 'code', 'source': ['dangerous_training()']}]}
    with pytest.raises(ValueError, match='export role'):
        list(exporter.export_cells(notebook))


def test_export_restores_recipe_without_restoring_other_machines_paths(exporter, tmp_path):
    namespace = {'REPO': REPO, 'DATA_ROOT': REPO / 'data/processed/d2c_v3',
                 'SPLITS': REPO / 'data/splits/splits_85_15.json',
                 'INPUT_MODE': 'diff_squares', 'BATCH_SIZE': 32,
                 'SQUARES_PER_PAIR': 16, 'PAIRS_PER_LOAD': 2,
                 'GATE_WITH_DIFF': True, 'NUM_WORKERS': 12, 'RESUME': None}
    saved = {'DATA_ROOT': '/other-machine/data', 'SPLITS': '/other-machine/split',
             'INPUT_MODE': 'whole_page', 'BATCH_SIZE': 4, 'GATE_WITH_DIFF': False,
             'PAIRS_PER_LOAD': 999, 'RESUME': '/other-machine/weights.pt'}
    exporter.apply_config(namespace, saved, data_root=tmp_path, num_workers=0)
    assert namespace['DATA_ROOT'] == tmp_path.resolve()
    assert namespace['INDEX'] == tmp_path.resolve() / 'master_dataset_with_benign.json'
    assert namespace['SPLITS'] == REPO / 'data/splits/splits_85_15.json'
    assert namespace['INPUT_MODE'] == 'whole_page'
    assert namespace['GATE_WITH_DIFF'] is False
    assert namespace['PAIRS_PER_LOAD'] == 4
    assert namespace['NUM_WORKERS'] == 0
    assert namespace['RESUME'] is None


def test_notebook_paths_work_from_repo_or_notebook_folder(exporter, monkeypatch):
    notebook = json.loads(exporter.NOTEBOOK.read_text(encoding='utf-8'))
    _, config = next(exporter.export_cells(notebook))
    monkeypatch.delenv('DATA_ROOT', raising=False)
    monkeypatch.delenv('FC_SIAM_RUNS_DIR', raising=False)
    for cwd in (REPO, exporter.NOTEBOOK.parent):
        monkeypatch.chdir(cwd)
        namespace = {}
        exec(config, namespace)
        assert namespace['REPO'] == REPO
        assert namespace['SPLITS'] == REPO / 'data/splits/splits_85_15.json'
        assert namespace['RUNS_DIR'] == REPO / 'outputs/fc_siam_diff/runs'


@pytest.mark.parametrize('height,width', [(32, 32), (33, 35)])
def test_legacy_import_preserves_model_class_and_spatial_output(height, width):
    torch = pytest.importorskip('torch')
    from regression_lens.models.fc_siam_diff import SiamUnet_diff

    legacy = load_script(REPO / 'models/fc_siam_diff/siamunet_diff.py')
    assert legacy.SiamUnet_diff is SiamUnet_diff
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = SiamUnet_diff(3, 14).eval()
        other = legacy.SiamUnet_diff(3, 14).eval()
        other.load_state_dict(model.state_dict(), strict=True)
        image = torch.randn(1, 3, height, width)
        with torch.inference_mode():
            result = model(image, image)
        assert result.shape == (1, 14, height, width)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result.exp().sum(1), torch.ones(1, height, width))
    finally:
        torch.set_num_threads(previous_threads)
