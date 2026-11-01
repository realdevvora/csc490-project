# ADR 0003: Source-page-disjoint splits

- **Status:** Accepted
- **Date:** 2026-10-06
- **Owners:** Regression Lens team
- **Supersedes:** None
- **Related:** [Data design](../data-design.md), [experiment protocol](../experiment-protocol.md)

## Context

Each Design2Code source page can produce a reference image, several mutations, and benign variants. If derivatives of one page appear in different partitions, a model can partially memorize its content, colors, and layout. Pair-level random splitting would therefore overstate generalization.

## Decision

Assign each source webpage to exactly one of train, validation, or internal test. Every reference, mutation, and benign variant derived from that source remains in the same partition.

The assignment must be stored in a versioned manifest created from a recorded seed. Generating new variants must not change the assigned partition of an existing source page.

## Rationale

The research question concerns generalization to unseen webpages. Source-page-disjoint splits align the internal evaluation unit with that question and block the most direct form of content/layout leakage.

## Alternatives considered

- **Random split by generated pair:** simple and statistically balanced, but leaks page identity across partitions.
- **Split by mutation family:** measures transfer to unseen operations, but does not prevent page memorization and answers a different question.
- **Time-based split:** useful only if acquisition time represents a meaningful distribution shift, which has not been established.

## Consequences

### Positive

- Internal metrics better reflect performance on unseen webpages.
- All models use a stable, auditable split.
- Adding variants does not create cross-partition derivatives.

### Negative / trade-offs

- Class and mutation-family counts may be less balanced than with pair-level splitting.
- The split builder must group before sampling and report per-page as well as per-pair counts.
- Results from incompatible earlier splits cannot be compared directly.

## Validation / evidence

The leakage mechanism follows directly from multiple highly correlated images sharing one source layout. The first dataset pilot should verify programmatically that every `source_page_id` maps to exactly one partition.

## Revisit when

- the research question changes to within-page monitoring rather than unseen-page generalization; or
- an additional evaluation explicitly targets transfer across mutation families and is kept separate from the primary page-disjoint evaluation.
