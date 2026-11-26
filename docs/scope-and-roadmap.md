# Scope and roadmap

This file separates commitments from ideas and preserves scope history. When direction changes, update the current sections and append a row to the history table; do not erase the old decision. Use an ADR when the change affects data compatibility, labels, splits, dataset roles, evaluation, or the primary research question.

## Current committed scope

- Investigate paired-image visual regression detection for webpages.
- Generate labelled reference/candidate pairs from Design2Code HTML using controlled mutations.
- Generate benign/no-change examples separately from regression mutations.
- Record mutation provenance, render configuration, localization, and source-page identity.
- Use source-page-disjoint train and validation splits (an internal test split is optional).
- Establish simple baselines before comparing larger candidate models.
- Compare models on DiffSpot's comparison half, and evaluate the frozen integrated system once on its final-evaluation half.
- Preserve enough configuration and environment information to reproduce serious experiments.

## Probable extensions

- nuisance-aware DINOv2 or ChangeFormer variants;
- pixel masks in addition to target bounding boxes;
- richer benign cross-browser or rasterization variations;
- human-labelled severity or acceptability;
- natural-language descriptions of detected regressions; and
- HTML-email evaluation if webpage results and time permit.

These are not required for minimum project success and should not shape the first pipeline unless promoted through an explicit scope decision.

## Explicit non-goals for the initial project

- training a large vision-language model from scratch;
- production SaaS deployment;
- Outlook/Gmail rendering farms;
- automatic browser-agent QA;
- a full Figma-to-HTML comparison product;
- a general-purpose mutation plugin marketplace;
- a database or distributed orchestration platform; and
- scaling to the full source dataset before the pilot is inspected and trusted.

## Change procedure

For a proposed scope or direction change:

1. State the current assumption and the proposed replacement.
2. Link the evidence: experiment, paper, benchmark limitation, schedule, or engineering constraint.
3. Describe effects on existing data, metadata, splits, and result comparability.
4. Create or supersede an ADR for consequential decisions.
5. Append the change below and update `project-status.md`; do not rewrite prior rows.

## Scope-change history

An earlier DINO research note considered DiffSpot adaptation. It remains an
unadopted proposal: accepted ADR 0005 forbids training on either DiffSpot half.
The discussed four-detector/VLM comparison has not replaced that ADR.

Full-page inference implementation choices (native-scale windows, overlap
aggregation, border treatment and scoring) are recorded in the
[full-page protocol](results/dino/fullpage-run.md). They implement the approved
experiment expansion below without changing labels, dataset roles or splits.

| Date | Change | Reason | ADR / issue |
|---|---|---|---|
| 2026-10-06 | Established the documented bootstrap scope: webpage pairs from Design2Code, source-page-disjoint internal evaluation, and DiffSpot-only external evaluation. | Create a reproducible baseline before data or model work expands. | ADR 0001–0003 |
| 2026-11-13 | DiffSpot is split into a model-comparison half and a final-evaluation half; an internal Design2Code test split is no longer required. | Design2Code validation can no longer separate the fine-tuned models, and the project needs both a held-out model comparison and an untouched evaluation of the integrated system. | ADR 0005 |
| 2026-11-14 | Prioritized a local, zero-shot DINOv2 patch-feature study with matched regression/benign/no-change crops, followed by identical DINOv3 inputs. Head training is deferred until feature sensitivity is understood. | General dataset validation is complete; establish DINO-specific evidence and explain intermediate tensors before training. | [Notebook guide](../notebooks/README.md); existing dataset/label/split ADRs remain unchanged |
| 2026-11-14 | Added a separately logged, ROI-centred position follow-up while preserving the frozen baseline crop and sample IDs. Future backbone comparisons must repeat both manifests. | The DOM-centred position crop did not contain the actual changed glyphs, so its zero response could not measure feature sensitivity. | [DINO pilot evidence](results/dino/README.md); shared dataset and label contracts unchanged |
| 2026-11-14 | Advanced the DINOv2 priority to full-page inference, a frozen-backbone detection/localization head, and a benign-data ablation while DINOv3 access is pending. GPU setup and throughput benchmarking precede larger runs. | The user requested results that night or the following morning; a lab RTX 4080 is available and local CPU extraction is too slow for full-page experiments. | [Project status](project-status.md), [lab setup](lab-setup.md); VLM integration remains a later group stage |
| 2026-11-14 | Locked six DINO configurations: A training-free, B frozen spatial head with benign supervision, C without benign supervision; both backbones, three trained seeds, unchanged 390/69 split. Deferred DiffSpot work for this implementation round. | User explicitly reaffirmed original experiment scope and exact repo splits, and approved implementation of steps 1–4. | [Frozen-head protocol](results/dino/head-runs.md); no changes to labels, exclusions or dataset roles |
| 2026-11-14 | Prepared a versioned crop correction with union-extent fallback only for zero-coverage anchors; preserved the running v1 B/C experiment. | Crop-specific audit found two anchors outside the reference extent. Only those two regression crops and one matched control change; no dataset labels/splits or A predictions change. | [Crop audit](results/dino/head-runs.md); v1/v2 results must remain separate |

| 2026-11-25 | Integrate DINO with main, organize notebooks by task/model and reconcile current documentation with ADR 0005. | Prepare one reviewable team checkout without changing experiments, split IDs or scoring semantics. | [Integration audit](integration-audit.md); ADR 0005/0006 unchanged |
| 2026-11-26 | Organize FC-Siam after its merge into main, preserve legacy Python entry points, and restore saved checkpoint settings during export. | Keep all four team tracks navigable and prevent notebook defaults from changing checkpoint inference. | [FC-Siam integration](fc-siam-integration.md); training recipe, labels, split and scorer unchanged |
