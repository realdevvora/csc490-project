# FC-Siam-diff compatibility entry points

The model is now part of the shared project layout. Start with the
[FC-Siam workflow](../../notebooks/fc_siam_diff/README.md).

- [Model implementation](../../src/regression_lens/models/fc_siam_diff.py)
- [Training notebook](../../notebooks/fc_siam_diff/train_fc_siam_diff.ipynb)
- [Feature specification](../../docs/models/fc_siam_diff/spec.md)
- [Technical analysis](../../docs/models/fc_siam_diff/technical_analysis.md)
- [Curated predictions, metrics and charts](../../docs/results/fc_siam_diff/README.md)

`save_predictions.py`, `plot_results.py` and the `siamunet_diff` import remain
compatibility entry points. Existing ignored `runs/` folders are left in place;
pass their checkpoint path to the exporter. New runs use
`outputs/fc_siam_diff/runs/`. The notebook and curated results have moved to
the links above; there are no duplicate notebooks or result copies here.

`requirements.txt` remains usable; the equivalent project extra is `fc-siam`.
The original model attribution and upstream licensing limitation are recorded
in the workflow and feature specification.
