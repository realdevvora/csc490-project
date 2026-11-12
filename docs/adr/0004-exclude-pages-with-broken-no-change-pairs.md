# ADR 0004: Exclude pages with a broken no-change pair

- **Status:** Accepted
- **Date:** 2026-11-11
- **Owners:** Regression Lens team
- **Supersedes:** None. Extends [ADR 0003](0003-source-page-disjoint-splits.md).
- **Related:** [Data design](../data-design.md), [ADR 0002](0002-regression-and-benign-label-policy.md), `data/splits/splits_85_15.json`, `scripts/make_team_split.py`

## Context

Every source page has one `no_change` pair: the page re-rendered under the same conditions, which should be pixel-identical to its reference. Models rely on these pairs to learn, and to show, that an unchanged page is not a regression. The no-change false-positive rate is measured on them.

25 of the 484 `no_change` pairs are not identical:
- 15 differ in height, by 1 to 30 px;
- the other 10 are the same size but have changed pixels, from 240 to 177,358.

The re-render is evidently not always deterministic. For example, fonts or images may load differently. This was found when a model with an exact-identity guarantee still flagged two validation `no_change` pairs: both turned out to differ from their reference.

These pairs are not valid no-change examples. They are not reliably benign either: some differences are visible content changes, such as a different colour. So relabelling them as `benign` would put unverified, possibly regression-like changes into the benign class.

## Decision

A page is excluded from every partition, along with all of its pairs, if its `no_change` pair is not pixel-identical. That means the reference and candidate differ in size, or `changed_px > 0`.

The team split (`data/splits/splits_85_15.json`) is regenerated as if those pages did not exist:
- the remaining pages are shuffled with seed 490 and the last 15% are held out for validation;
- the excluded pages are listed under the split file's `"excluded"` key;
- `scripts/make_team_split.py` reproduces the file.

The rule is implemented in `regression_lens.data.splits.broken_no_change_pages`.

## Rationale

- Excluding the whole page keeps every remaining page complete: one reference, its mutations, its benign variants and a valid no-change pair. Per-page counts stay consistent.
- Dropping only the broken pair would leave 25 pages without a no-change example.
- A deterministic, metadata-only rule is auditable and can be rerun after the dataset is regenerated.

## Alternatives considered

- **Relabel the broken pairs as `benign`:** rejected. Some differences are content changes, such as colour, not rendering noise. Under ADR 0002, benign examples must come from an explicit, identified process.
- **Drop only the broken `no_change` pairs:** keeps 304 more pairs, but leaves those pages without a no-change example. It also keeps the pages' other pairs, which come from the same non-deterministic render.
- **Regenerate the broken renders:** the best long-term fix, but it changes the dataset, and nothing guarantees the re-render will be stable. It can be done later, as noted under Revisit when.
- **Keep the pairs:** rejected. It penalises a model for flagging a real difference, and it understates the no-change false-positive rate of a model that treats identical inputs exactly.

## Consequences

### Positive

- No-change metrics measure only true identity.
- No unverified differences enter the benign or no-change classes.
- The exclusion is listed in the split file and reproduced by a script, so it can be audited.

### Negative / trade-offs

- 329 of 7,271 pairs (4.5%) are no longer used: 254 regression, 50 benign and 25 no-change.
- The team split changes from 411/73 to 390/69 pages. Only 17 of the new validation pages were in the old validation set, so results on the old split are not comparable and need a rerun.
- The rule checks metadata (`size_ref`, `size_cand`, `changed_px`). It relies on those fields being accurate.

## Validation / evidence

- The metadata rule flags 25 pages.
- A pixel-by-pixel comparison of all 484 `no_change` pairs flags exactly the same 25.
- `tests/test_splits.py` checks that the committed split matches `make_team_split` on the local dataset.

## Revisit when

- the dataset is regenerated with deterministic re-renders, so that no pages fail the rule; or
- a benign class for same-browser rendering noise is defined explicitly, with human review of each pair.
