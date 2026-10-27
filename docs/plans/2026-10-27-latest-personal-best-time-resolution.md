# Latest User-Reported Personal-Best Time Resolution

> Keep this resolver opt-in. It may replace a conflicting value only from a later, explicit, user-authored personal-best statement for the same event.

**Goal:** Use the latest user-reported personal-best time when older and newer cited snapshots conflict.

**Architecture:** For questions asking the user's current personal best, scan only validated user-authored quotes. Require a personal-best phrase, an exact `minutes:seconds` or `minutes and seconds` value, matching event terms, one unambiguous observation date no later than the question date, and a unique value on the latest date. Ignore assistant text, previous-best questions, conflicting values on the same date, future observations, and unrelated events. Replace the answer and evidence only after the candidate passes the full answer validator. The CLI flag `--latest-personal-best-time-resolution` is off by default.

## Evidence

Stage243 question `6a1eabeb` asks for the personal-best time in a charity 5K. The saved answer cites 27:12 from a May 23 memory. A later user-authored May 30 statement identifies 25:50 as the personal best while discussing another charity 5K; it is within the June 25 question date. The answer generator used the older value.

## Verification and outcome

- Adapter suite: 140 passed, 25 subtests passed; `compileall` and `git diff --check` passed.
- Frozen replay on Stage243's 500 saved answers changed only `6a1eabeb`, from 27:12 to 25:50. The new citation and answer passed validation; the judge returned PASS with seed 42 accepted. Score moved 438/500→439/500, with one candidate-only correction and no baseline-only regression.
- The same one-row correction occurred on the Stage252 saved 500-answer candidate, 428/500→429/500.
- Tests also verify that assistant-only repetitions, ambiguous same-date values, and questions about a previous personal best do not trigger the resolver.

## Decision

Keep the resolver opt-in. It uses an explicit later user statement, not an assistant paraphrase. These are saved-answer replays, not new generations; Stage243 remains the best fresh full Top-20 result at 438/500 (87.6%). The 97% and 100% memory-precision goals remain unmet.

Adapter commit: `63a3ea9 feat: resolve latest user-reported personal-best time`. Scorecard: `docs/operations/data/longmemeval-stage255-latest-personal-best-time-pilot-2026-10-27.json`.
