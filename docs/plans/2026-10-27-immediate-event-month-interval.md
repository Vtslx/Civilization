# Bounded Immediate-Event Month Intervals

> Keep this resolver opt-in. It may compute an event-month answer only when user-authored citations bound every relevant date and every possible date pair rounds to the same month count.

**Goal:** Recover month-interval answers when one event is described as just completed but its exact day is omitted from the memory.

**Architecture:** Reuse the existing citation-checked immediate-event interval builder. For an explicitly cited `just completed` or `just finished` event with one unambiguous observation date, bound the event to the six days before that observation through the observation date. Match events against the two clauses in a `between ... and ...` question. Compute the month count across every possible date pair using a 30.4375-day average month; rewrite only if all pairs yield the same rounded integer. Leave `event_date` unset and record the derived bounds in `date_basis`.

**Safety and compatibility:** Assistant-authored, weakly matched, missing-date, conflicting-date, overlapping-event, or month-boundary-ambiguous evidence cannot resolve the answer. The CLI flag `--temporal-immediate-event-month-resolution` is off by default. The existing event-list resolver keeps its original set of accepted immediate-event phrases.

## Evidence

Stage243 question `gpt4_21adecb5` asks for months between completing an undergraduate degree and submitting a master's thesis. The user cited “just completed” in a conversation observed on 2022-11-17; the thesis submission was explicitly “today” on 2023-05-15. The first event's exact day is unknown, but its six-day interval (2022-11-11 through 2022-11-17) yields six months against every possible day for the later event. The baseline abstained and was judged wrong.

## Verification and outcome

- Adapter suite: 123 passed, 25 subtests passed; `compileall` and `git diff --check` passed.
- Frozen Stage243 temporal replay (133 rows): the resolver returned a result on two rows and changed one answer. `gpt4_21adecb5` changed from an abstention to `6 months`; citation validation passed and the configured judge returned PASS. No saved correct answer regressed; the other resolved row's answer was unchanged.
- Full 500-question candidate: 428/500 (85.6%) versus Stage243's 438/500 (87.6%). The stored Top-20 order matched Stage243 in 500/500 rows; answerer and judge accepted seed 42; all rows completed with zero unresolved API errors. Paired outcomes were 19 candidate-only, 29 Stage243-only, 409 both-correct, and 43 both-wrong (net −10). The feature triggered only for `gpt4_21adecb5`, which was corrected in the candidate. The overall full-run decline reflects regenerated-answer/provider variation and is not attributed to this one-row postprocessor.

## Decision

Keep the narrowly bounded resolver opt-in based on the positive frozen-answer correction with zero answer regressions. Do not claim a full-score improvement: Stage243 remains the best verified full Top-20 score at 438/500 (87.6%), and the 97% target and 100% memory precision remain unmet.

Adapter commit: `db72119 feat: resolve bounded immediate event month intervals`. Scorecard: `docs/operations/data/longmemeval-stage252-immediate-month-interval-pilot-2026-10-27.json`.
