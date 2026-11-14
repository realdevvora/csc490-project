# Project status

**Last updated:** 2026-11-14

**Phase:** matched DINOv2/DINOv3 feature study / GPU experiment expansion

**Latest checkpoint:** DINOv3 access and local authentication now work, and
checkpoint 4 completed on the font-size regression/benign/no-change trio.
Benign top-5%-mean cosine distance is 0.116949 versus regression 0.0261631;
no-change L2 is exactly zero. The comparison figure and raw statistics were
reviewed; see [pilot notes](results/dino/README.md). Next extend DINOv3 to the
39 frozen baseline cases plus the separate position trio: this expanded run has
now completed locally, saving 84 summaries and 14 comparison figures. DINOv3
benign scores exceed regression in 11/13 baseline trios (DINOv2: 12/13);
width and border-width have stronger regression than benign responses.
All no-change L2 maxima are zero. No trained head or full-page runner exists yet.

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
authentication and the expanded matched-sample run still need verification.

## Next milestones

1. Transfer the completed matched DINOv3 pilot and updated source to the already configured lab workspace; review selected figures alongside raw statistics.
2. Implement and benchmark label-blind full-page DINOv2 inference, then a frozen-backbone spatial head for detection/localization and a benign-training ablation. These trained-head/full-page jobs are not implemented or running yet.
3. Prioritize one completed run of each DINOv2 condition, then repeat trained runs with additional seeds; measure GPU throughput before promising deadline completion.
4. Repeat any implemented DINOv2 training conditions with DINOv3 under matched settings. DiffSpot evaluation and the later group VLM stage remain separate.

The crops are guided by known mutation locations. These observations study local feature sensitivity and must not be reported as full-page detection performance. The sample/crop manifest and local artifacts live under ignored `outputs/dino_fit/`. See the [notebook guide](../notebooks/README.md).
