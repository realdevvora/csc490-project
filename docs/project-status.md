# Project status

Reader guide: [Understanding experiments A, B and C](dino-experiments-explained.md).

**Last updated:** 2026-11-14

**Phase:** A/B/C v1 core comparison complete / artifacts audited and results documented

The user reported the completed A/B/C table: DINOv2+B leads validation AUROC at
0.9485 ± 0.0092, versus DINOv3+B at 0.9145 ± 0.0110. Without benign supervision,
C is near chance for both backbones. Pixel difference still has the strongest
pointing accuracy. See [the reported results and interpretation](results/dino/trained-validation-results.md).
The imported report bundle was inspected: all 12 completion records and 36
checkpoint prediction files pass compatibility/coverage checks, and recomputed
metrics match saved values. All four conditions select update 2,000; validation
AUROC is still improving through the saved checkpoints. Preserve v1 artifacts;
there is no urgent rerun requirement and no external-generalization claim.
A crop-specific audit found
two of 4,739 regression anchors have zero coverage because changed border strips
lie beyond the reference extent. They are excluded from positive sampling, not
mislabeled positives. A tested local v2 correction changes only these anchors
and one matched control, in a fresh run namespace; do not mix v1/v2 results or
restart solely for this small omission. See [crop audit and handling](results/dino/head-runs.md).

**Earlier checkpoints:** DINOv3 access and local authentication now work, and
checkpoint 4 completed on the font-size regression/benign/no-change trio.
Benign top-5%-mean cosine distance is 0.116949 versus regression 0.0261631;
no-change L2 is exactly zero. The comparison figure and raw statistics were
reviewed; see [pilot notes](results/dino/README.md). Next extend DINOv3 to the
39 frozen baseline cases plus the separate position trio: this expanded run has
now completed locally, saving 84 summaries and 14 comparison figures. DINOv3
benign scores exceed regression in 11/13 baseline trios (DINOv2: 12/13);
width and border-width have stronger regression than benign responses.
All no-change L2 maxima are zero. A full-page pretrained runner is now implemented
with label-blind windows, shared-coordinate maps, and a pixel baseline; see
[run policy and lab commands](results/dino/fullpage-run.md). Both lab smoke runs
completed 15 pairs on the RTX 4080: DINOv2 33.19 s, DINOv3 20.30 s, with peak
PyTorch allocation below 0.5 GiB each. Smoke prediction/identity checks passed.
Full validation scoring has now been reported for all 1,033 pairs: regression/
benign AUROC is 0.082 (DINOv2), 0.206 (DINOv3), and 0.097 (pixel); all three
have benign FPR@95R of 1.0. The shared F1 of 0.923 hides complete benign
false-positive failure at the validation-selected threshold. Pixel pointing
accuracy is 0.953 versus 0.539/0.585 for DINOv2/v3. See the
[baseline interpretation](results/dino/fullpage-run.md). A frozen-head implementation now exists for
conditions B/C, with three seeds, bounded sampled-window caches, shared full-page
evaluation and validation checkpoint selection. See [head run instructions](results/dino/head-runs.md).
Local synthetic tests do not establish dataset accuracy; reported real-data
head results are now linked above. The user confirmed that both 15-pair GPU smoke
runs passed prediction-coverage and identity checks, and launched full validation.
DiffSpot work is deferred by user request.

Before lab experiments, follow the recorded [DINO findings and experiment
controls](results/dino/experiment-controls.md): label-blind full-page coverage
first, matched frozen-head comparisons and benign-data ablations next. The note
also lists implementation choices that remain open.

## What exists

- project, architecture, data, scope, and ADR documentation;
- shared metadata loading, a committed 390/69 train/validation page split, and the 25-page exclusion rule in ADR 0004;
- shared prediction JSONL format, scoring code, and tests;
- model investigations on separate team branches; and
- three executed DINOv2 checkpoints and an executed first-trio DINOv3 comparison in `notebooks/dino_exploration.ipynb`. A standalone matched DINOv3 pilot runner now repeats the original and supplementary manifests without resampling; see [run instructions](results/dino/matched-pilot-run.md).

The team has completed general dataset validation and manual inspection. DINOv2 has now run locally on all 39 fixed baseline pair/crop cases (approximately 218 seconds), plus three separately recorded position follow-up cases. Width gives the clearest selected response, small text-color change is weak, and benign browser responses usually dominate raw distances. All no-change L2 maps are zero. The original position crop missed the changed content; a supplementary crop reveals a response and is kept distinct from the baseline. These are exploratory examples, not operator-wide or full-page results. See the [pilot notes](results/dino/README.md). No head is trained. A shared DiffSpot adapter is still pending.

## Confirmed decisions

- The primary domain is web interfaces.
- Design2Code-derived pairs are used for training and internal evaluation.
- DiffSpot remains an independent external generalization benchmark.
- Intentional visible CSS/DOM mutations are regressions.
- Benign/no-change examples are generated separately.
- Splits are disjoint by source webpage.
- No final model has been selected.

## Open questions

- Which mutation examples produce concentrated versus diffuse DINO patch-feature responses?
- How do regression distances compare with matched benign and no-change controls?
- How much do the 14 px versus 16 px patch sizes affect localization?
- How do results change across more pages and mutation magnitudes per operator?
- How should later full-page inference cover changed content far from a DOM element's centre?
- Does DINOv3 improve nuisance separation and localization across the matched examples?

## Blockers

No dataset-location or DINOv3 access blocker remains. Local data are under
`D:\csc409-data\d2c_v3`; approved access was verified by actual model inference.

The user has connected to a lab host and reported an RTX 4080 (16 GB), NVIDIA
driver 580.178.04, Ubuntu 22.04.5, and Python 3.10.12. Remote GPU environment
installation and project/data transfer have completed. The host
currently requires interactive password authentication; no password is stored in
the repository or shared with the agent. See [GPU lab setup](lab-setup.md).

The home quota issue was resolved by using `/virtual/.cache/$USER/csc490` for
the environment, project, data, and caches. CUDA-enabled PyTorch 2.8.0+cu128
was verified on the RTX 4080 and the DINOv2 first trio ran successfully.
Use explicit environment `python3` commands on the lab host. Remote DINOv3
authentication and both backbone smoke runs have been verified.

## Next milestones

The user has now requested research on a full detector/localizer/VLM system and
future DiffSpot train/validation/held-out-test adaptation. Recommendations and
the resulting protocol distinction are in the [generalization/VLM roadmap](research/generalization-and-vlm-roadmap.md).
No new training, DiffSpot split or VLM integration has been launched.

1. Preserve completed experiment A predictions, raw metrics and maps; review the recorded benign-rejection failure and pixel localization baseline.
2. Preserve selected head weights, manifests and original v1 reports on the lab host; the imported bundle deliberately excludes weights/raw maps.
3. Use the verified report and summary figure for the six-configuration comparison, retaining the validation-selection and crop-coverage caveats.
4. Any longer-training or localization revision must be a separately declared follow-up. DiffSpot and VLM work remain deferred.

The crops are guided by known mutation locations. These observations study local feature sensitivity and must not be reported as full-page detection performance. The sample/crop manifest and local artifacts live under ignored `outputs/dino_fit/`. See the [notebook guide](../notebooks/README.md).
