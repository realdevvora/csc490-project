# Architecture Decision Records

Architecture Decision Records (ADRs) preserve consequential research and engineering decisions. They complement current-state documentation by recording why a choice was made and when it should be reconsidered.

## Convention

- Use `NNNN-short-decision-name.md` with the next available number.
- Never renumber or silently rewrite an accepted ADR.
- Correct minor errors in place, but supersede an ADR when the decision changes materially.
- Link supporting experiments, issues, pull requests, and documentation when available.
- Update current-state documents after accepting or superseding a decision.

## Statuses

- **Proposed:** under discussion and not yet binding.
- **Accepted:** the current project decision.
- **Superseded:** replaced by a later ADR; retain it as history.
- **Rejected:** considered but not adopted.

## Index

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-training-and-external-evaluation-datasets.md) | Training and external-evaluation datasets | Accepted |
| [0002](0002-regression-and-benign-label-policy.md) | Regression and benign label policy | Accepted |
| [0003](0003-source-page-disjoint-splits.md) | Source-page-disjoint splits | Accepted |

## Template

```markdown
# ADR XXXX: <Decision title>

- **Status:** Proposed | Accepted | Superseded | Rejected
- **Date:** YYYY-MM-DD
- **Owners:** <names or team>
- **Supersedes:** <ADR number if applicable>
- **Related:** <issues, PRs, experiments, docs>

## Context

What problem, ambiguity, or project change required a decision?

## Decision

What are we deciding?

## Rationale

Why did we choose this option?

## Alternatives considered

What other options were considered and why were they not selected?

## Consequences

### Positive

- ...

### Negative / trade-offs

- ...

## Validation / evidence

What experiment, paper, benchmark, engineering constraint, or observation supports this decision?

## Revisit when

What future condition would justify reconsidering this decision?
```
