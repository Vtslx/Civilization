# Composite Aggregate Evidence Coverage

> Keep this answer guard opt-in. If a numeric total joins two components with “and”, every component must have a validated count item tied to user-authored evidence.

**Goal:** Prevent a complete-looking total when retrieval supports only one side of a multi-entity question.

**Architecture:** Limit the gate to numeric aggregate questions with an explicit total cue, one conjunction, and a structured count-item audit. Extract distinct target terms on each side. A component is covered only when a validated `CountedItem` has a user-authored source message and an exact citation that identifies the component. If any component is missing, preserve partial item provenance, clear the aggregate `count_value`, and replace the answer with an abstention. Assistant text cannot establish coverage. Comma-separated enumerations, non-total questions, and questions without a structured count audit remain unchanged. The CLI flag `--composite-count-evidence-coverage` is off by default.

## Evidence and acceptance

Stage243 had two wrong abstention rows where the answerer returned an unsupported total even though its audit listed only one component:

- `edced276_abs`: evidence supported 10 days in Hawaii but no Seattle duration.
- `e5ba910e_abs`: evidence supported $378 for headphones but no iPad cost.

The pilot must check all saved answers for this shape, preserve complete totals, require exact user-authored citations, validate every changed abstention, and have no baseline-correct regressions.

## Verification and outcome

- Adapter suite: 131 passed, 25 subtests passed; `compileall` and `git diff --check` passed.
- Stage243 saved-answer replay examined 500 rows. Fifteen entered the simple two-component gate; thirteen with complete component evidence stayed unchanged. The two missing-component totals became abstentions. Both passed answer validation and the judge. The replay moved 438/500→440/500, with 2 candidate-only, 0 baseline-only, 438 both-correct, and 60 both-wrong.
- The same guard applied to the saved Stage252 full answer set changed those same two incorrect answers and no others, moving 428/500→430/500. This reused existing generation outputs and rejudged only changed answers; it is not a fresh generation run.

## Decision

Keep the default-off evidence-coverage guard. It corrected two false-positive totals without changing complete audited totals. Stage243 remains the best fresh full Top-20 run at 438/500 (87.6%); this replay does not establish the 97% target or 100% memory precision.

Adapter commit: `5f6a8ac feat: abstain on unsupported composite totals`. Scorecard: `docs/operations/data/longmemeval-stage253-composite-total-evidence-coverage-pilot-2026-10-27.json`.
