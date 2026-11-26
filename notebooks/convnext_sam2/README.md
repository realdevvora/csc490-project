# ConvNeXt and SAM2 encoder experiments

Zehao Fan / Philip's notebooks, moved from `src/zehao/` during integration.

- **Current comparison:** `CSC490_ConvNeXt_SAM2_team.ipynb` reports the committed
  390/69-page split. ConvNeXt A means trained without benign; DINO A is training-free.
- **Historical only:** `ConvNeXt.ipynb`, `SAM2.ipynb` generate their own 411/73
  split. Do not use those runs for current team comparisons or selection.

Colab/Drive setup needs a GPU, torch, torchvision, timm, NumPy, Pillow,
scikit-learn, pandas and matplotlib. This is not a portable environment lock.
Configure Drive paths and copy the committed split unchanged. The extracted
`/content/d2c_v2` directory name does not identify the dataset version.
Never commit datasets, tokens or weights; preserve versioned run namespaces.

SAM2 Hiera-Tiny is used as a Siamese classification encoder, not a promptable
mask decoder. Exported top-tile box/centre predictions need the shared scorer.
Quick notebook metrics are rounded and use quantile FPR95/fixed-.5 thresholds;
no shared-scorer localization result is verified here. The team scoring example
now uses the real repository URL and explicit data root. Training code and
outputs are preserved. See [model directory](../../docs/model-index.md) and
[integration audit](../../docs/integration-audit.md).
