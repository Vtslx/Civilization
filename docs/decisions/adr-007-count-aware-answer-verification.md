# ADR-007: Require a complete audit for numeric answer revisions

**Status:** Accepted for an opt-in pilot
**Date:** 2026-10-26

## Context

LongMemEval Stage232's Top-20 evidence covered all answer-marked pairs in 453/470 answerable questions, yet 47 incorrect rows still had complete pair coverage. Numeric answer synthesis remains a major gap. The existing citation-backed verifier returns only an answer string and citations; changing a numeric total without changing the associated `count_value`, item inventory, and temporal audit fails the full validator. The rejected Stage237 count adjudicator also showed that reconciling multiple free-form drafts is not reliable by itself.

## Decision

- Add an opt-in numeric verification contract that returns an answer, citations, a complete item inventory, a total/unit, and temporal events when required.
- Run the existing validation pipeline over a candidate copy. For counts, require exact citations, distinct qualifying items, compatible units, matching item totals, and a numeric answer equal to `count_value`.
- Preserve the original structured answer intact when the audit is missing, invalid, or inconsistent.
- Keep the feature disabled by default. The frozen 94-question pilot must show positive paired net gain before committing or recommending a full run.

## Consequences

- Numeric verification adds one model call for each selected question type and can increase latency.
- Complete audit fields make accepted changes reviewable and preserve a deterministic rollback path.
- The prompt and model may still miss evidence; pilot results must be paired on the same frozen answers/evidence, and a full 500-question run remains necessary before any release score claim.

## Rejected alternatives

- **Accept a revised numeric answer without updating its audit:** breaks count consistency and permits unsupported values.
- **Parse a number from the verifier's answer string:** loses evidence for individual items, time windows, and user-authorship constraints.
- **Enable a second verifier globally:** previous temporal second-pass experiments regressed correct answers; keep this narrowly opt-in.
