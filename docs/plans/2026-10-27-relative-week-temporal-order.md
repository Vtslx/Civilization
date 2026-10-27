# Conservative Relative-Week Temporal Order Implementation Plan

> **For Codex:** Execute the steps in order in the Civilization and `memory-benchmarks` checkouts. Keep the change conservative and preserve the current answer whenever date intervals overlap or provenance is incomplete.

**Goal:** Resolve temporal comparisons that use a user-authored “last week” event date when cited evidence proves a unique order.

**Architecture:** Extend the existing citation-backed relative-date interval parser used by `resolve_temporal_choice_order`. Anchor “last week” or “previous week” to one unambiguous memory observation date and represent it with a deliberately broad interval covering both a rolling-week and prior-calendar-week interpretation. The existing ordering resolver changes an answer only when the intervals are disjoint. No new model call or second retrieval pass is added.

**Tech Stack:** Python, Pydantic temporal evidence, existing observation-date and citation validators, LongMemEval frozen scorecards.

---

## Evidence and scope

Stage241 question `gpt4_b4a80587` asks which happened first: a road trip or arrival of a new lens. The model's event audit has exact user citations but records “last week” without a normalized date interval for the road trip; the lens quote “a month ago” resolves to 2023-04-23. The existing choice-order resolver therefore preserves a wrong model answer. The original Stage241 answer/evidence and Stage221 retrieval snapshot are frozen inputs for a paired 133-question temporal pilot.

Only user-authored “last week” and “previous week” phrases with an unambiguous observation date are in scope. Interpret the phrase as the broad interval from 13 days before the observation date through 1 day before it. This covers rolling-week and previous-calendar-week readings. Do not resolve an order if this interval overlaps any other event interval. Do not add “last month,” weekdays, seasons, or assistant-authored relative dates.

## Invariants and acceptance gates

- Date bounds come only from an exact user-authored quote and a single observation date.
- Assistant-only, unstructured, missing-date, and conflicting-date evidence remains unresolved.
- The existing resolver must preserve the original answer when event intervals overlap.
- Stage242 replays all 133 Stage241 temporal answers with frozen selected Top-20 evidence and reruns judging only for changed answers.
- Keep the code only if the frozen pilot has at least one candidate-only correction and zero baseline-only regressions. Otherwise revert the parser change.
- After a positive pilot, run a full 500-question candidate from the Stage221 retrieval snapshot; keep the feature opt-in if it does not improve the best full score.

## Task 1: Add failing parser and resolver tests

**Files:** `memory-benchmarks/tests/test_longmemeval_prompts.py`

1. Reproduce `gpt4_b4a80587` with user-authored “last week” and “a month ago” citations; assert the existing resolver currently leaves the incorrect order unchanged.
2. Assert disjoint relative intervals resolve to the earlier lens event.
3. Assert an overlapping “last week” and “ten days ago” pair stays unchanged.
4. Assert assistant-only quotes and missing observation dates stay unresolved.

Run the focused tests and verify the positive case fails before implementing the parser.

## Task 2: Extend the relative interval parser

**Files:** `memory-benchmarks/benchmarks/longmemeval/prompts.py`

Recognize only `last week` and `previous week` in a validated user citation. Bound the phrase to `[observation_date - 13 days, observation_date - 1 day]`, intersect it with other supported citations for the event, and return unresolved if the intersection is empty. Reuse `_explicitly_user_authored_quote`, `_memory_observation_date`, and the existing disjoint-interval gate.

Run focused tests, then `python3 -m pytest tests/test_longmemeval_prompts.py -q`, the full adapter suite, `python3 -m compileall benchmarks tests`, and `git diff --check`.

## Task 3: Run the frozen Stage242 temporal pilot

Use all 133 Stage241 temporal answers and their selected Top-20 IDs; reconstruct full memory text from the Stage221 snapshot and preserve the answer audit. Run the choice-order resolver only, judge changed answers with the configured gateway, and report paired outcomes. Do not rerank or regenerate answers in this isolated pilot.

**Stage242 result:** Adapter commit `f455c0d fix: resolve cited last-week temporal order` passed 111 tests and 22 subtests; `compileall` and `git diff --check` passed. The frozen pilot changed one answer (`gpt4_b4a80587`), which moved from incorrect to correct. Temporal accuracy moved 114/133→115/133; paired outcomes were 1 candidate-only, 0 baseline-only, 114 both-correct, and 18 both-wrong. The frozen full diagnostic moved 433/500→434/500; this is not a fresh full score. Artifact: `docs/operations/data/longmemeval-stage242-last-week-temporal-frozen-pilot-2026-10-27.json`.

The existing N-event list-order resolver was also replayed on the frozen temporal answers after the parser change. It produced no additional answer changes or new list resolutions.

## Task 4: Commit the code and validate the full candidate

Stage242 passed its isolated gate and the adapter change is committed. Stage243 originally
stopped after 253 completed answers because the local Magpie auto group routed the alias to
a Home GPT-6 model that rejected Chat Completions. With user authorization, the Home member
was disabled for `group/auto-gpt-6-luna`; the group ID and Codex member remain enabled. The
change repaired the route but converted this auto-discovered group to user-managed state.

The full Stage243 run resumed over all 500 frozen Stage221 retrieval records. It used
Top-200 retrieval with Top-20 answer context, the Stage241 reranker allowlist, multi-session
answer verification, `skip_multi_session` count inventory policy, temporal event-list
ordering, and seed 42. Three saved rows with empty judge responses were retried using their
existing generated answers; 2 passed and 1 failed, with answer text unchanged.

**Stage243 result:** 438/500 (87.6%) at Top-20, 47 below the 485-answer gate. Against
Stage241 (433/500), paired outcomes were 21 candidate-only, 16 baseline-only, 417 both
correct, and 46 both wrong (+5 net). The full per-type counts are knowledge-update 72/78,
multi-session 106/133, single-session-assistant 51/56, single-session-preference 27/30,
single-session-user 68/70, and temporal-reasoning 114/133. Top-20 source-session recall was
470/470; 871/883 annotated answer-message pairs (98.6%) were retrieved, and 47 answers were
still wrong despite all annotated pairs appearing in Top-20. API errors: zero. This paired
delta includes routing and model variability; it does not isolate the relative-week resolver.
Stage243 did not measure Top-200 answer accuracy or the separate 100% memory-precision goal.

Validated scorecard: `docs/operations/data/longmemeval-stage243-relative-week-top20-2026-10-27.json`.
The candidate does not meet 97%; retain temporal resolvers as opt-in and use the per-type and
pair-coverage diagnostics to choose the next experiment. Do not treat a Top-200 diagnostic as
the Top-20 release gate.
