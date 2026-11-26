"""Compatibility entry point; use scripts/export_fc_siam_predictions.py."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                       'scripts/export_fc_siam_predictions.py'), run_name='__main__')
