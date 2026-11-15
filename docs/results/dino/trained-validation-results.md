# DINO A/B/C: verified v1 validation comparison

Initially recorded 2026-11-14 from the lab comparison table, then verified against
the imported `outputs/lab-results/dino-results.tar.gz` bundle. The table retains
rounded values; detailed metrics and learning logs were inspected locally.
These are the original **v1 crop-manifest runs**, not corrected v2 runs.

## Completed comparison

All configurations use the committed 69 validation pages (1,033 pairs). B/C use
three seeds, 490/491/492, with validation-selected common checkpoint steps per
condition/backbone. AUROC variability is sample standard deviation across seeds.

| Backbone / baseline | Condition | Regression/benign AUROC | Seed SD | Benign FPR@95R | Pointing accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| DINOv2 | A: training-free distances | 0.0819 | — | 1.0000 | 0.5387 |
| DINOv3 | A: training-free distances | 0.2060 | — | 1.0000 | 0.5847 |
| Pixel difference | Pixel | 0.0970 | — | 1.0000 | 0.9528 |
| DINOv2 | B: with benign training | 0.9485 | 0.0092 | 0.2633 | 0.6505 |
| DINOv2 | C: without benign training | 0.4663 | 0.0027 | 1.0000 | 0.6457 |
| DINOv3 | B: with benign training | 0.9145 | 0.0110 | 0.4130 | 0.5985 |
| DINOv3 | C: without benign training | 0.4609 | 0.0013 | 1.0000 | 0.6146 |

For B/C, AUROC, FPR and pointing values are means over seeds. A/pixel values are
single training-free runs, not three-seed means. The supplied DINOv3 selection
log selects step 2,000 for both B and C. Inspection confirms that DINOv2 also
selects step 2,000 for both conditions.

## Artifact and metric verification

- All 12 training completion records report 2,000 updates and checkpoints at
  500/1,000/2,000. Each learning log contains 81 finite diagnostic records from
  update 1 through 2,000. Recorded configurations match the fixed batch size,
  learning rate, weight decay, condition and seed.
- All training configurations contain the same v1 window manifest: 11,818
  windows on the exact 390 training pages. Manifest dataset/split hashes match
  local data and the committed split; window source IDs belong to training.
- Both evaluations cover the exact same 1,033 validation IDs and 18 heads each.
  All 36 prediction files pass shared format/coverage checks. Recomputing shared
  metrics from predictions at the saved thresholds reproduces all saved AUROC,
  detection/FPR, localization and recall-breakdown values within 1e-12.
- Recomputing the declared selection rule confirms step 2,000 for all four
  backbone/condition combinations. The bundle omits checkpoint tensors and raw
  map archives by design, so this audit does not independently inspect learned
  weights or rerun model inference.

Reproducible local analysis is in
[analyze_v1.py](../../../outputs/lab-results/analyze_v1.py), with
[audit JSON](../../../outputs/lab-results/analysis-v1/audit.json) and a
[summary figure](../../../outputs/lab-results/analysis-v1/validation_summary.png).
These local artifacts are ignored by Git. Original reports and review PNGs are
under `outputs/lab-results/extracted-v1/outputs/dino_heads/`.

## Learning progress and operating points

| Condition | AUROC at 500 | AUROC at 1,000 | AUROC at 2,000 |
| --- | ---: | ---: | ---: |
| DINOv2 B | 0.8704 | 0.9242 | 0.9485 |
| DINOv2 C | 0.3853 | 0.4370 | 0.4663 |
| DINOv3 B | 0.7477 | 0.8589 | 0.9145 |
| DINOv3 C | 0.3421 | 0.4136 | 0.4609 |

Fixed-training losses decrease and all fixed-training diagnostic AUROCs end at
1.0, including C. That diagnostic excludes benign pairs and uses windows, so
it cannot establish benign rejection or full-page generalization. C's lower
training loss is not evidence of a better model; its task lacks hard negatives.
Learning plots show noisy batch losses and finite gradient spikes, not numerical
divergence in the logged samples.

There is no observed validation-AUROC downturn at the three saved checkpoints.
Both B models are still improving through the budget, particularly DINOv3.
This neither rules out overfitting nor establishes convergence/underfitting.
The backbone ranking is conditional on this fixed 2,000-update recipe. A longer
matched B-only comparison would be a new experiment, not a replacement for these
results; no urgent rerun is needed to complete the current core comparison.

| Selected condition | Mean recall at max-F1 threshold | Mean benign FPR at max-F1 threshold | Mean benign FPR@95R |
| --- | ---: | ---: | ---: |
| DINOv2 B | 97.13% | 35.27% | 26.33% |
| DINOv3 B | 96.61% | 46.14% | 41.30% |

The max-F1 operating point accepts more false alarms to catch more regressions
on this class mix. Do not confuse it with the separate 95%-recall operating
point. All selected heads have zero no-change FPR at their max-F1 threshold.

DINOv2 B per-seed AUROCs are 0.9517/0.9380/0.9556; DINOv3 B values are
0.9031/0.9153/0.9252 (seeds 490/491/492). The observed ordering holds across
these seeds, but they are not independent validation datasets.

## Error patterns worth retaining

Per-operator AUROC here means that operator's regressions versus all 138 benign
pairs, averaged over seeds at the selected checkpoint:

- DINOv2 B is strongest on font-size (0.9833), width (0.9756) and background-colour
  (0.9756), and weakest on opacity (0.8731) and text-colour (0.9031).
- DINOv3 B is weakest on border-radius (0.7583; only 25 regression examples),
  opacity (0.7750) and text-colour (0.8364). Letter-spacing and margin are stronger.
- At max-F1 thresholds, mean benign Firefox/WebKit FPRs are 31.40%/39.13% for
  DINOv2 B and 45.89%/46.38% for DINOv3 B. These browsers were represented during
  training, so this is not an unseen-browser transfer experiment.
- In the predeclared seed-490 examples, small-change pointing is 44.57% for both
  B models (175 regressions, changed-box area <=0.1% of reference image area),
  compared with about 75% for large changes (>1%). This is single-seed descriptive
  evidence and geometric box area, not precise changed-pixel area.

Inspected learning plots for both backbones and two review examples:
`d2c_481_webkit` is a confident DINOv2 B false positive with browser/layout text
variation; `d2c_054_border_radius` is a low-confidence DINOv3 B false negative
whose localization peak falls in the broad target region. These illustrate that
detection confidence and localization maps can disagree. Border/padding response
is a plausible failure mechanism in the figures, not a proven dataset-wide cause.

C also saturates scores: for seed 490, all 138 benign scores equal 1.0, as do
775/826 DINOv2 and 763/826 DINOv3 regression scores. Many tied maxima limit ranking
and support the interpretation that identity-only training learns an overconfident
change detector. Unsaturated logits were not exported; do not invent recovered
rankings or change the reported score definition retrospectively.

## Conditional source-page uncertainty check

A paired bootstrap resamples the 69 source pages, keeping each page's related
pairs together, with 2,000 replicates and seed 490. For each replicate it averages
AUROC across the existing three trained seeds. Approximate 95% percentile
intervals for AUROC differences are:

| Difference | Interval |
| --- | --- |
| DINOv2 B minus DINOv3 B | [0.0045, 0.0607] |
| DINOv2 B minus DINOv2 C | [0.4502, 0.5061] |
| DINOv3 B minus DINOv3 C | [0.4214, 0.4824] |

These are conditional on the selected checkpoints. Selection is not repeated
inside bootstrap samples, and the same validation set was already used for
selection. They describe sensitivity to validation-page composition; they do
not remove selection optimism or establish independent external generalization.

## Main finding: benign supervision changes detection substantially

B versus C is the most direct controlled comparison: same backbone/head recipe,
update budget, positive sampling stream and validation data; the negative mix
changes from no-change-only to benign plus no-change.

- DINOv2 AUROC rises from 0.4663 to 0.9485: an absolute increase of 0.4822.
- DINOv3 AUROC rises from 0.4609 to 0.9145: an absolute increase of 0.4536.
- At 95% regression recall, mean benign false-positive rate drops from 100% to
  26.33% for DINOv2 and 41.30% for DINOv3.

This supports the value of the implemented benign-supervision recipe on these
validation pages. A head trained only against identity controls does not learn
useful regression/benign ranking here. It does not prove that frozen DINO learns
a universal notion of harmlessness or intent.

B versus A shows an overall learned-pipeline improvement. It does not isolate
the head alone: A uses top-5% pooled raw distances, while B/C use learned window
classification with maximum-window aggregation. B versus C is the cleaner
ablation for the contribution of benign supervision.

## Leading configuration and unresolved localization

DINOv2+B leads this validation comparison: AUROC 0.9485 versus 0.9145 for
DINOv3+B, with lower benign FPR@95R (0.2633 versus 0.4130). The AUROC difference
is 0.0340. This is a validation ranking, not a statistical significance claim
or evidence of superiority on a new dataset.

The learned heads do not beat pixel pointing accuracy (0.9528). DINOv2+B reaches
0.6505 and DINOv3+B reaches 0.5985. Within a backbone, B versus C changes pointing
by only +0.0048 for DINOv2 and -0.0161 for DINOv3. Thus the clear gain from benign
supervision is detection/rejection, not a consistent localization improvement.

Pointing is scored on regressions independently of the detection threshold.
For non-reflow cases its target is pixel-difference-derived, which partly
explains the direct pixel baseline's geometric advantage. Large/partial weak
boxes, patch resolution, and localization-loss behavior require inspection;
this table alone does not establish why the heads miss a location.

## Limits and next analysis

- The same validation pages select checkpoints and decision thresholds. There
  is no independent internal-test result. Seed SD is training variability, not
  an uncertainty interval for new webpages; many pairs share a source page.
- A benign FPR of 26.33% at 95% recall is a meaningful improvement but still
  produces substantial nuisance alarms. AUROC alone does not establish an
  acceptable operating threshold. Inspect precision/recall and negative FPRs
  at each validation-selected threshold.
- Selecting the final saved step does not show that training has converged,
  nor prove underfitting. The inspected trends are described above.
- The v1 manifest omits positive coverage in two regression anchor crops out of
  4,739. Preserve and disclose this small limitation. The prepared correction
  changes those crops and one matched control; do not mix versions or discard
  these results solely for that omission.
- The imported bundle passes the record/prediction checks above. Checkpoint
  weights and raw maps remain on the lab machine; preserve them with the
  manifests before any cleanup. This report does not authorize another run.
- DiffSpot remains deferred. Do not claim external generalization or begin
  another tuning round merely to improve the reported validation comparison.

See [the student guide](../../dino-experiments-explained.md),
[the baseline protocol](fullpage-run.md), and [the training protocol](head-runs.md).
