# FC-Siam-diff validation evidence

Dev's run `20261111-210302` (run 3, best checkpoint at pass 33), imported by
PR #8. The prediction file, saved metrics and five charts are preserved from
that merge. This directory contains curated experiment evidence, not weights
or a dataset. Full details: [technical analysis](../../models/fc_siam_diff/technical_analysis.md)
and [training/export workflow](../../../notebooks/fc_siam_diff/README.md).

| Shared-scorer metric | Value |
|---|---:|
| Validation pairs | 1,033 |
| Regression / benign AUROC | 0.7250 |
| Benign FPR at 95% regression recall | 0.9493 |
| Detection precision / recall / F1 | 0.8568 / 1.0000 / 0.9229 |
| Benign / no-change FPR | 1.0000 / 0.0000 |
| Type macro-F1 (826 regressions) | 0.2921 |
| Pointing (729 supported regressions) | 0.9369 |
| Mean box IoU (729 supported regressions) | 0.9166 |

- [Complete saved scorer output](fc_siam_diff_val.metrics.json)
- [Original comparison table](comparison.md)
- [Predictions](fc_siam_diff_val.jsonl)
- [Score distributions](plots/scores.png), [ROC](plots/roc.png),
  [per-type detection](plots/detection_by_type.png), [type F1](plots/type_f1.png),
  [type confusion](plots/confusion.png)

The fixed team split has 69 validation pages: 826 regressions, 138 benign
browser pairs and 69 no-change pairs. The detection threshold (approximately
0.08158) maximizes F1 on these validation predictions. It is not a threshold
validated on an untouched external test set.

Localization is **conditional on an emitted point/box**: 729 of 826 regression
pairs (88.3%) have these outputs; the shared scorer omits the remaining 97
from its localization means. DINO's box results cover all 826 regressions,
so the IoUs and pointing values cannot be compared as equal-coverage metrics.
The FC-Siam pipeline also gates predictions with raw pixel differences.
The checkpoint was selected by validation pixel F1; other tracks may select by
AUROC. These results describe one trained run, with no multi-seed uncertainty
estimate.

During the 2026-11-26 integration, the saved predictions were rescored with
the unchanged shared scorer against the local D2C metadata and team split.
The saved metrics were reproduced; no retraining or threshold-rule change
was involved. Historical source paths inside the saved JSON are retained as
provenance even though the evidence now lives under `docs/results/`.
