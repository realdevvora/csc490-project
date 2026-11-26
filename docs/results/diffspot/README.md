# DiffSpot preparation evidence, not model results

The [aggregate audit](preparation-audit.json) records metadata from all 4,400
records of HF revision `c6dd79d5e1c0cbb4e7ca234c9f53c418a75e30ce`.
Only annotation columns were read for this release-wide check. Four real image
pairs were separately prepared as a preliminary smoke test. No detector, Kimi
or judge was run; these counts do not measure benchmark performance.

- 3,900 changed pairs: 1,300 per tier, 300 per native operator.
- 500 no-diff controls; this revision has no unknown-difficulty controls.
  The pinned official loader's historical unknown-control compatibility is retained.
- Exact property-aware type mapping: 2,616 mapped / 1,284 unmapped changed pairs.
  All remain in detection/localization/description evaluation.
- 330 target-element boxes extend beyond the 1280x800 viewport; none have an
  empty visible intersection. Source boxes remain intact, with visible boxes
  used only for supplementary localization metrics.

Full `prepare` reproduces counts and mapping coverage and retains native
properties in `annotations.jsonl`. Aggregate its `gt_mutations[*].property`
and `source_target_box` to reproduce this metadata audit. Data/annotations and
smoke outputs stay ignored; only these aggregate counts are committed evidence.
See [workflow](../../diffspot-benchmark.md) for commands and evaluation contracts.
