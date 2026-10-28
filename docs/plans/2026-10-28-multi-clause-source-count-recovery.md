# Multi-Clause Count Recovery from User Evidence

## Goal

Recover a missing start or current value in a two-part count answer when the existing answer audit is valid but contains only one of the requested state snapshots. Keep the existing opt-in `--multi-clause-count-answer-synthesis` boundary.

## Approaches considered

1. Keep requiring two answerer-produced count items. This is safest but cannot repair the observed Stage269 error: a valid one-item audit returned only the current value although Top-20 contained both user statements.
2. Ask the model to regenerate the missing count. This adds another stochastic model decision and weakens source-level auditability.
3. Extract both values from exact user-authored sentences. Chosen: it is deterministic, ties each value to a direct user citation, and can abstain when either state, unit, action, role, or count is ambiguous.

## Design

The fallback runs only for a numeric count question with exactly two clauses classified as start and current, the same simple unit and action in both clauses, one existing counted item, and a base answer that already passes the full validator. It scans only explicitly marked user messages. Each state needs a sentence with a number immediately associated with the requested unit, the same action, and the named role context when present. The start state must have a start marker; the current state must have a current marker. Conflicting start counts or conflicting latest current counts leave the answer unchanged. The current snapshot is selected by the chronological memory index supplied to answer generation.

On success, build two `cumulative_total` items with exact citations, set `count_value` to the current value, and answer both clauses. The standard count and citation validator remains the final gate. Assistant text, unmarked memories, missing counts, unsupported actions, and incompatible units cannot establish either value.

## Acceptance and verification

- Recover the observed `031748ae` answer from user-authored 4-engineer start and 5-engineer current statements.
- Reject conflicting start values without mutating the original answer.
- Keep the feature disabled unless its existing CLI flag is enabled.
- Validate the exact citations and resulting count audit, replay all 500 saved answers, live-judge changed answers with seed 42, and run a full 500-question candidate over frozen retrieval before recording a scorecard.
