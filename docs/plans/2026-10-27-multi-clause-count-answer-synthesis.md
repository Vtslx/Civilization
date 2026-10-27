# Multi-Clause Count Answer Synthesis

> Keep this answer formatter opt-in. It must preserve separately requested count values without aggregating or guessing them.

**Goal:** Answer a two-part start/current count question with both cited values when the structured audit already contains each fact.

**Architecture:** Apply only to numeric count questions containing exactly two “how many” clauses, one explicitly about the start/initial state and one about the current state. Require exactly two distinct counted items, user-authored citation validation, compatible units, a unique start/current state mapping, and lexical overlap between each clause and its item. Construct a concise labeled answer in question order, use the current-state value for `count_value`, then validate the full candidate. Ambiguous mappings or invalid candidates leave the original answer unchanged. The CLI flag `--multi-clause-count-answer-synthesis` is off by default.

## Evidence

Stage243 knowledge-update question `031748ae` asks how many engineers the user led when starting a Senior Software Engineer role and how many they lead now. Its audit already held two user-cited, validated snapshots: 4 at role start and 5 in the latest update. The generated answer was only “5 engineers,” so the two-part answer was judged wrong.

## Verification and outcome

- Adapter suite: 135 passed, 25 subtests passed; `compileall` and `git diff --check` passed.
- Frozen replay over Stage243's 500 saved answers changed only `031748ae`. The candidate answer “At the start: 4 engineers; now: 5 engineers.” passed validation and the judge. No previously correct answer changed, moving the saved-answer replay from 438/500 to 439/500.
- The same resolver on Stage252's 500 saved answers changed only that question and passed the judge. This moves that saved-answer replay from 428/500 to 429/500.
- Combined with Stage253's two abstention corrections, the saved-answer replay totals are 441/500 from Stage243 and 431/500 from Stage252. These are postprocessor replays, not fresh answer-generation runs.

## Decision

Keep the multi-clause formatter opt-in. It preserves two independently cited values and corrected one answer with no regressions. Stage243 remains the best fresh full Top-20 run at 438/500 (87.6%); neither the 97% score target nor 100% memory precision is achieved.

Adapter commit: `d8bfd4f feat: preserve multi-clause count answers`. Scorecard: `docs/operations/data/longmemeval-stage254-multi-clause-count-answer-synthesis-pilot-2026-10-27.json`.
