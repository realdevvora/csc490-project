"""Compatibility import for existing FC-Siam notebooks and checkpoints."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from regression_lens.models.fc_siam_diff import SiamUnet_diff

__all__ = ['SiamUnet_diff']
