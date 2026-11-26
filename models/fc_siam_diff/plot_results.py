"""Compatibility entry point; use scripts/plot_fc_siam_results.py."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
                       'scripts/plot_fc_siam_results.py'), run_name='__main__')
