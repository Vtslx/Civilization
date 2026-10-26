# ADR-002: Resolve temporal intervals only from cited user events

- Status: Accepted for the LongMemEval answer adapter
- Date: 2026-10-26

## Context

Stage225 reached 430/500, with 56 of 70 wrong answers despite all answer-marked pairs appearing in Top-20. Auditing temporal failures showed two cases the current answerer did not resolve reliably: it substituted the question date for an explicit event anchor introduced by “when I …”, and it abstained on a trip-duration question despite cited user-authored start and return events.

## Decision

Add deterministic post-generation temporal resolution in the memory-benchmark answer adapter. A date may be used only when supported by an exact user-authored citation. Relative phrases such as “today” are anchored to the cited memory’s observation date. Trip duration requires distinct, matching user-authored start and return events with unambiguous dates. On missing or conflicting evidence, keep the answer unresolved.

## Alternatives considered

1. Add another model call to re-evaluate temporal answers. Rejected because it adds latency and may overrule correct citations without guaranteeing date consistency.
2. Infer the event date from the question date when an explicit anchor is missing. Rejected because the question can refer to a prior event date that differs from the question date.
3. Resolve only from citation-backed user text and observation metadata. Chosen because the computation is auditable, deterministic, and can abstain safely when the timeline is incomplete.

## Consequences

- The answer adapter may correct event-relative intervals without changing stored memory or retrieval ranking.
- Assistant-only or ambiguous event dates cannot trigger a correction.
- The implementation must be benchmarked against the fixed 500-question Top-20 protocol; targeted examples are diagnostic only.
