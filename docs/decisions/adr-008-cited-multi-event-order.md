# ADR-008: Rewrite multi-event order only from disjoint cited date intervals

**Status:** Positive isolated pilot; full candidate ties the best score; remain opt-in
**Date:** 2026-10-27

## Context

Stage239 retained 97.96% of answer-marked pairs in Top-20 and had full pair coverage in 455/470 answerable questions. Among 70 wrong answerable rows, 61 already had all marked pairs in Top-20. Several temporal list-order answers included only dated events or omitted an event whose position could not be derived. Retrieval coverage alone does not repair chronology or event-list completeness.

## Decision

- Add an opt-in N-event order resolver for temporal list questions.
- Require each listed event to have a validated user-authored citation and a source-derived exact date or bounded date interval.
- Resolve the list only when the expected event count matches the evidence events, descriptions are distinct, and all date intervals are pairwise disjoint.
- Rebuild the ordered answer from the cited event descriptions and preserve the citations. If any date, event identity, or list completeness check is ambiguous, keep the model answer unchanged.
- Pilot on Stage239's frozen temporal answers and selected Top-20 evidence; do not regenerate answers or reranker selections during the isolated replay.

## Consequences

- The resolver can correct ordering when event dates are clear while leaving ambiguous events untouched.
- It does not infer an order from conversational mention order, vague phrases such as “recently,” or overlapping intervals.
- The behavior remains opt-in until the paired frozen pilot shows corrections without regressions and a later full 500-question candidate supports it.

## Rejected alternatives

- **Sort all event strings unconditionally:** can promote unsupported or duplicate events and invent an order for missing dates.
- **Ask the model to answer the list again:** provider variance would change both extraction and answer synthesis, obscuring the resolver effect.
- **Use only the event date emitted by the model:** dates must be corroborated by exact supplied user evidence before they control output order.
