# Generalization, localization and VLM integration: research roadmap

**2026-11-14. Status: research recommendations, not a launched experiment.**
The user now proposes DiffSpot adaptation on one partition and evaluation on
another. This changes its earlier external-only role. Record a superseding ADR
and fixed source-group manifests before adaptation; do not silently reinterpret
the completed Design2Code results. No DiffSpot data was downloaded, partitioned,
trained on or scored during this research pass. Existing 390/69 splits stay intact.

## Evidence and task mismatch

Our [verified v1 experiments](../results/dino/trained-validation-results.md) show
that benign supervision is essential to the implemented detection recipe.
DINOv2+B leads, but its max-F1 operating point still flags 35.3% of benign pairs.
Pixel pointing is much stronger than learned pointing; small-change localization
is particularly weak. DINOv3 B is still improving at 2,000 updates, so preserve
it as a candidate rather than treating the current ranking as universal.

DiffSpot's release has 4,400 pairs: 3,900 changed and 500 no-diff. The released
partition is named `test`, not an official adaptation split. Its task requires
describing the visual difference, not just producing a positive binary score.
[Official dataset card](https://huggingface.co/datasets/tencent/DiffSpot).

The paper deliberately isolates single-property localized changes and filters
reflow contamination. Its operators include text, gradients and justification;
they do not map one-to-one onto our mutation names. Consequently it does not
fully test multi-defect pages or browser-nuisance rejection.
[DiffSpot paper](https://arxiv.org/html/2605.29615v1).

The official evaluator judges free-form answers against structured mutation
records; it reports tier recall, no-diff specificity and overall case accuracy.
Pin the evaluator and judge version. High Design2Code detection AUROC is not
equivalent to this description-level score.
[Evaluation protocol](https://github.com/Tencent/DiffSpot/blob/master/docs/evaluation.md).

## Proposed system: propose, verify, ground, explain

```text
Expected screenshot + actual screenshot
                   |
        Size/viewport consistency checks
                   |
   Semantic features + pixel/edge evidence
                   |
     High-recall candidate region proposals
                   |
  Region scoring + fine localization refinement
                   |
  Paired crops + page context + optional OCR
                   |
     VLM verifies and describes differences
                   |
 Grounded issues, original-coordinate boxes, confidence/review state
```

Start with image-region interfaces rather than directly wiring DINO embeddings
into a language model. Existing VLMs cannot consume arbitrary DINO vectors
without a trained projector/interface. OneDiff's explicit visual-delta module
is useful research for a later learned connector, not a free interchangeability
guarantee. [OneDiff](https://arxiv.org/abs/2407.05645).

Our current absolute-difference/product/cosine paired representation is symmetric:
swapping reference and candidate leaves the comparison features unchanged. This
is suitable for change presence but cannot encode which direction a property
changed. Preserve ordered expected/actual images for the VLM; any future numeric
direction-prediction head must also receive ordered information.

Change captioning research also supports treating localization, nuisance
discrimination and description as related but distinct problems.
[Robust Change Captioning](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Robust_Change_Captioning_ICCV_2019_paper.html).

### First priority: improve candidate coverage and localization

- Retain DINOv2+B as the semantic baseline. Add raw RGB differences, gradients
  and fine-scale visual evidence for candidate generation/refinement. Do not
  merely interpolate a coarse map and call it higher-resolution information.
- Union proposals from semantic and pixel evidence: requiring both to agree
  could discard precisely the subtle colour/text changes we need to recover.
- Produce several nonduplicate regions, with context margins and paired
  before/after coordinates. Initially compare budgets of 3/5/10 regions on
  validation and measure changed-target coverage, proposal IoU and VLM cost.
- Preserve global context with a page overview. Do not feed only a magnified
  tiny crop when the issue concerns alignment or relationships between elements.
- Test a lightweight high-resolution decoder/shallow pixel branch before a
  wholesale backbone replacement. Multi-scale Siamese designs are relevant
  architectural precedents, but satellite-trained ChangeFormer weights are not
  validated UI weights. [ChangeFormer](https://arxiv.org/abs/2201.01293).
- Use exact synthetic changed-pixel masks where available as *visible-change*
  supervision. Keep edited-element localization distinct from propagated reflow
  effects; the current weak boxes do not justify pixel-perfect mask claims.
- Handle input-size discrepancies explicitly. Permit only justified global
  registration; local warping can erase a genuine position/layout regression.

Do not hard-gate VLM access using the current max-F1 detector threshold. Tune
proposal recall and final reporting confidence separately. Evaluate misses from
proposal generation, verification and description separately; candidate misses
cannot be fixed by a better crop-only VLM downstream.

### Second priority: generalization data and training

- Maintain benign and no-change replay when adapting on DiffSpot. Do not undo
  B's main lesson by adapting only to changed versus identical pairs.
- Add regression-plus-browser-variation examples: current isolated regression
  and isolated nuisance examples do not establish performance on their mixture.
- Add negative screenshots from more rendering conditions and hard false
  positives mined on training pages only. Define which rendering differences
  count as acceptable before generating labels; browser provenance alone is
  not a universal acceptability rule.
- Sample more training windows over time instead of two permanently fixed
  windows. Cover targets and background at several context sizes, preserving
  native-resolution detail and exact coordinate transforms.
- Do not independently colour-jitter the pair or label arbitrary text/style
  differences benign. Such augmentation could destroy the positive label.
- Add text/gradient/alignment/removal and compound-change coverage deliberately
  as new dataset versions. Preserve the original baseline dataset and reports.
- Run a bounded, matched longer-training B comparison for v2/v3 before deciding
  whether partial backbone adaptation is needed. Later unfreezing/LoRA should
  use small learning rates, synthetic replay and fixed validation selection.

## VLM shortlist and integration experiment

| Candidate | Proposed role | Primary evidence |
| --- | --- | --- |
| Qwen3-VL-4B-Instruct | First implementation/pilot; image-region verification and description | [Model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) documents GUI, OCR and spatial capabilities |
| Qwen3.5-4B | Newer same-size comparison; measure rather than assume improvement | [Model card](https://huggingface.co/Qwen/Qwen3.5-4B) provides multimodal inference; needs a newer software stack |
| InternVL3_5-4B | Optional independent-family control | [Model card](https://huggingface.co/OpenGVLab/InternVL3_5-4B) includes multi-image examples and visual grounding |

These are candidates, not established DiffSpot winners. Begin with Qwen3-VL-4B
and one alternative; avoid a large model sweep. Official Qwen training code
supports multi-image data and freezing its vision component:
[fine-tuning framework](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md).

For the RTX 4080, profile a 4B model with limited visual tokens before committing
to adaptation. Four billion FP16 parameters alone are approximately 8 GB; vision
tokens, caches, buffers and gradients require more. An 8B FP16 model already
uses about 16 GB for weights alone. Quantization and LoRA/QLoRA are experiments
with their own quality/memory tradeoffs, not guaranteed fits. Use a separate VLM
environment and run stages sequentially rather than breaking the DINO environment.

Provide unmarked expected/actual crops plus an overview. Use overlays as optional
extra views, not replacements that obscure pixels. Prompt the VLM to verify
whether a proposed region actually differs; a proposal is not ground truth.
Return region ID, affected element/text, visible property, expected-vs-actual
direction, supported description and a review/uncertain state. Prefer externally
tracked region coordinates over unconstrained invented coordinates.

The model should say "the label looks larger" when supported, not invent a CSS
selector, exact CSS declaration, severity or root cause from screenshots alone.
An approved design is the reference contract; images cannot independently reveal
whether an unannounced difference was intentional.

Pilot frozen VLMs first. Then adapt only the best candidate with LoRA if error
analysis identifies a learnable description/verification gap. Use realistic
predicted crops, partial crops and no-change/nuisance negatives in training,
not just oracle target crops or repeated templated operator names. Metadata can
supply training targets but must not enter inference prompts or filenames shown
to the model. Human-check generated training descriptions on training data only.

## DiffSpot protocol and credible comparisons

1. Freeze a source-page-grouped train/validation/test manifest *before* model or
   prompt selection. A provisional 60/20/20 allocation is reasonable, with group-
   constrained operator/tier/no-diff balance; actual sizes depend on source
   groups. Never random-split pairs from the same page across partitions.
2. Verify real page/source identifiers in the release. `domain` is a category,
   not proof of a unique source. If IDs are insufficient, use exact reference
   image hashes and near-duplicate grouping; disclose the grouping limitation.
   Check overlaps with Design2Code and keep connected source groups together.
3. Preserve the original Design2Code 390/69 split. Use DiffSpot training for
   adaptation; its validation partition selects prompts, budgets, thresholds and
   checkpoints. Lock its test partition and record all attempted configurations.
4. On that identical held-out subset compare: VLM-only; pixel proposals + VLM;
   DINO proposals + VLM; hybrid proposals + VLM. Use the same VLM/adaptation
   access and measure token/latency budgets. Add GT-box + VLM only as an explicitly
   labelled oracle diagnostic, never as a deployable headline result.
5. Measure proposal recall/IoU, verification FPR/recall, description correctness,
   final boxes/points, no-diff specificity, operator/tier results, runtime and
   paired source-page uncertainty. Count end-to-end misses including gated-out
   changed examples. Retain a separate cross-browser challenge set.
6. Pin official evaluator/prompt/judge versions. If adding crop tools or custom
   structured output, report a modified/assisted protocol and compare a fair
   baseline under it. Audit a predeclared sample of judge decisions for unsupported
   claims and wrong-element descriptions. Report abstention/coverage separately.
7. Adapted held-out DiffSpot scores are not directly comparable to its published
   full-set zero-shot leaderboard. Reevaluate baselines on the same held-out
   subset, using its actual denominators. Do not call adaptation an untouched
   external-generalization evaluation. Retain another untouched source/domain
   challenge if that is a claim we want to make.

The separate [WUICC research](https://arxiv.org/abs/2607.01728) is relevant to a
future UI-description evaluation beyond one benchmark. It is not an additional
required dataset or an automatically comparable score in this proposal.

## Recommended next milestone

Build the inference-only region-proposal + paired-crop VLM pilot using existing
Design2Code partitions and a locked prompt. Compare VLM-only, pixel-guided and
hybrid-guided variants before adapting anything on DiffSpot. Diagnose proposal
coverage versus description failure. Then finalize the source-disjoint DiffSpot
adaptation protocol and train only the stages whose failures justify training.
This makes the next step useful for the intended two-screenshot product, not
just a longer run of the current binary head.
