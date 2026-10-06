# ADR 0002: Regression and benign label policy

- **Status:** Accepted
- **Date:** 2026-10-06
- **Owners:** Regression Lens team
- **Supersedes:** None
- **Related:** [Data design](../data-design.md)

## Context

Early mutation ideas mixed mutation magnitude with semantic acceptability and an exploratory notebook randomly labels some intentional CSS changes as benign. That approach can make identical operations receive contradictory labels and can teach a model that deliberate design violations are acceptable.

## Decision

Label an intentional, visible CSS/DOM mutation as a regression. Generate benign examples through a separate, explicit process and identify their `variation_type`.

A small numerical mutation is not automatically benign. Severity scores, tolerance bands, and human-labelled acceptability are outside the initial binary label contract.

## Rationale

The generator knows that an intentional mutation departs from the source design, but it does not know whether a user or designer would tolerate that departure. Keeping regression mutations and benign rendering variations separate yields a clear, reproducible initial label definition.

## Alternatives considered

- **Use magnitude thresholds:** easy to automate, but CSS impact depends on context and small changes may still violate the design.
- **Randomly label milder mutations as benign:** creates label noise without semantic justification.
- **Human-label every generated pair immediately:** potentially stronger semantics, but too costly before the generator and taxonomy are stable.
- **Use only regression pairs:** simplifies generation but does not train or measure nuisance tolerance.

## Consequences

### Positive

- Label generation is deterministic and explainable.
- Benign-data quality can be studied independently.
- Later severity or acceptability labels can be layered on without redefining the initial mutation provenance.

### Negative / trade-offs

- The initial binary policy may classify visually negligible intentional mutations as regressions.
- Realistic benign variations require separate design and validation.
- The dataset does not initially capture subjective tolerance.

## Validation / evidence

The initial evidence is conceptual consistency: mutation intent is observable from generation provenance, while acceptability is not. Pilot-pair inspection and false-positive analysis on separately generated benign examples will test whether this label contract supports the research goal.

## Revisit when

- the team introduces a documented human-annotation protocol;
- a validated severity/acceptability definition is available; or
- experiments show the binary intent-based contract prevents useful nuisance modelling.
