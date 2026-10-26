# Conservative Immediate-Event Temporal List Order Implementation Plan

> **For Codex:** Execute in the current Civilization and `memory-benchmarks` checkouts. Keep the new behavior separately opt-in until the frozen pilot has been judged.

**Goal:** Order complete temporal event lists when user-authored evidence says an event just occurred but does not state an exact event date.

**Architecture:** Add a separate opt-in path to the existing N-event list resolver. For a cited user quote with a completed immediate-event phrase and an unambiguous observation date, derive a broad event interval from six days before observation through the observation date. Require meaningful token overlap between the event description and the exact quote. The existing all-events, completeness, and pairwise-disjoint interval gates remain authoritative.

**Tech Stack:** Python, Pydantic temporal evidence, existing LongMemEval citation/observation validators, frozen Top-20 scorecards.

---

## Evidence and scope

Stage241 qid `gpt4_d6585ce8` asks for five concerts and musical events in chronological order. Three events have explicit dates; two do not. Their selected Top-20 evidence includes user-authored “just got back from a music festival in Brooklyn” with observation date 2023-04-01 and “just saw them live with Adam Lambert at the Prudential Center” with observation date 2023-04-15. These can be bounded between the exact March 25 and April 8 events. Stage241 omitted their positions because the structured audit did not provide normalized event dates.

Only list-order resolution is in scope. Do not use assistant-authored statements, plans, uncertain or conflicting observation dates, weak description matches, or event-duration arithmetic. The interval `[observation_date - 6 days, observation_date]` intentionally includes “just” ambiguity. If any event intervals overlap, keep the original answer. The feature remains disabled by default.

## Invariants and acceptance gates

- Every inferred interval must come from an exact user-authored quote already cited for that event.
- The source memory must have exactly one observation date.
- Event description and quote must share at least two meaningful terms after stop-word removal.
- The observation interval is a bound, not an exact event date; no duration answer may consume it as an exact date.
- The N-event resolver still requires the complete expected list and pairwise disjoint intervals.
- Replaying all 133 Stage241 temporal answers must change no answer other than those newly resolved by this feature.
- Keep the code opt-in until changed answers are rejudged on the configured provider and a full 500-question candidate is available.

## Task 1: Add failing parser and integration tests

**Files:** `memory-benchmarks/tests/test_longmemeval_prompts.py`, `memory-benchmarks/tests/test_longmemeval_run.py`

1. Reproduce the five-event concert question with three exact-date events and two user-cited immediate-event quotes; assert the default resolver leaves the incomplete chronology unchanged.
2. Assert the opt-in resolver orders the five events as Billie Eilish, outdoor concert, Brooklyn festival, jazz night, Queen/Adam Lambert.
3. Assert assistant-only, missing observation date, weak event-description overlap, and conflicting observations leave the answer unchanged.
4. Assert overlapping intervals leave the answer unchanged and default-off behavior remains unchanged.

Run the focused tests first and verify the opt-in positive case fails before implementation.

## Task 2: Implement and wire the opt-in interval resolver

**Files:** `memory-benchmarks/benchmarks/longmemeval/prompts.py`, `memory-benchmarks/benchmarks/longmemeval/run.py`, `memory-benchmarks/tests/test_longmemeval_prompts.py`, `memory-benchmarks/tests/test_longmemeval_run.py`

Add `allow_immediate_event_intervals=False` to `resolve_temporal_event_list_order`. Only when enabled, derive `[observation_date - 6 days, observation_date]` from validated user citations containing “just got back,” “just came back,” “just returned,” “just attended,” or “just saw.” Require at least two meaningful description/quote token matches. Add CLI flag `--temporal-recent-event-order-resolution`, persist it in run metadata, and thread it through the answer-generation path. Keep the flag off by default.

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`.

## Task 3: Replay the frozen temporal answers

Reconstruct Stage241's selected Top-20 memory text from the Stage221 snapshot and replay only the temporal list-order postprocessor over all 133 temporal answers. Record changed qids and candidate text. The provider currently times out on Chat Completions; do not claim a judged pilot or enable the feature by default until the provider can rejudge changed answers.

**Stage244 offline result:** On all 133 frozen temporal answers, the opt-in path changed one answer, qid `gpt4_d6585ce8`. It inserted the Brooklyn festival and Queen/Adam Lambert concert between their exact neighboring events, matching the five event identities and order in the benchmark reference. No live judge was available, so this is an unjudged candidate, not a score. Artifact: `docs/operations/data/longmemeval-stage244-immediate-event-order-offline-replay-2026-10-27.json`.

## Task 4: Judge and run a full candidate after inference recovers

The feature is implemented and committed as `8292b8d feat: add opt-in immediate event ordering`, with the full test suite passing (114 tests, 25 subtests). It remains disabled by default. After provider inference recovers, rejudge only changed answers against their frozen Stage241 baseline. Require at least one candidate-only correction and zero baseline-only regressions before running a full 500-question candidate over the frozen Stage221 retrieval inputs. Record judged pilot and full-run evidence in Civilization with English documentation commits.
