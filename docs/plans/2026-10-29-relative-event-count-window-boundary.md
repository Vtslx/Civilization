# Relative Event Count Window Boundary Resolution Plan

> **For Codex:** Implement this plan task by task. Keep the rule default-off, evidence-backed, and committed as one adapter module after its frozen replay is accepted.

**Goal:** Correct numeric event counts that include a user-cited event exactly on the exclusive lower boundary of a relative “past N days/weeks” window.

**Architecture:** Add a deterministic, opt-in post-processor to the LongMemEval answer pipeline. It will parse only bounded day/week windows, confirm the question date and exact event date, independently corroborate a weekday-derived date against the user quote and source observation date, remove only one-count incremental events exactly on the lower boundary, recompute the numeric answer from the remaining items, and run the existing full validator before accepting the revision. Ambiguous dates, unsupported sources, interval dates, cumulative rows, and events inside the window remain unchanged.

**Tech Stack:** Python 3.10+, existing LongMemEval `CountedItem`/citation models and validators, pytest, saved Civilization retrieval snapshots.

---

## Evidence and alternatives

Stage294 row `88432d0a` asks “How many times did I bake something in the past two weeks?” on 2023-05-30. The audit includes five distinct events, including a sourdough bake dated 2023-05-16 from a user quote saying it was baked on Tuesday in a conversation observed 2023-05-21. That event is exactly 14 days before the question date. The LongMemEval answer is 4; Stage291, Stage293, and Stage294 all fail (6, 6, and 5 respectively). Stage294 has already deduplicated the repeated birthday cake and its remaining 5-item count differs from the label only by the boundary sourdough event.

Approaches considered:

1. **Prompt-only clarification:** tell the model that “past N days” excludes the exact boundary date. This is cheap but relies on another stochastic answer generation and does not validate the event date.
2. **Narrow deterministic boundary resolver (recommended):** inspect validated item-level dates after count review; exclude only a user-cited incremental event whose exact date equals `question_date - N days/weeks`; recalculate and validate the answer. This makes the boundary rule auditable and leaves all other items unchanged.
3. **General temporal interval rewrite:** recompute every event window and classify all dates. This is broader than the evidence and risks changing existing relative-month/year semantics.

The resolver will use option 2. The lower bound is open for “past N days/weeks”: the valid interval is `(question_date - N, question_date]`. A weekday-derived event date is eligible only when its exact user quote and observation date yield that date unambiguously. If any required piece is ambiguous, the original answer is retained.

## Task 1: Add regression and boundary tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test the exact 14-day boundary, a date one day inside the window, a date outside it, missing or conflicting observation dates, invalid/assistant-only citations, cumulative or multi-quantity rows, non-count questions, default-off behavior, and pipeline/category scope. The target fixture must derive 2023-05-16 from a user quote saying “Tuesday” in a source observed 2023-05-21, then revise 5 to 4 while preserving all other item citations.

## Task 2: Implement opt-in boundary reconciliation

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--relative-event-count-window-boundary-resolution`, off by default. Restrict it to `how many times` count questions with an explicit “past/last/previous N days/weeks” phrase, a parseable question date, and incremental event-count items. Independently derive the date only from an exact user citation plus a parseable source observation date. Remove a boundary item only if its quantity is exactly one and the cited date agrees with its item `event_date`. Rebuild the answer/evidence from retained items, validate it, and record a per-row audit with boundary date and excluded source indices. Plumb the flag through normal and evaluate-only execution and run metadata.

## Task 3: Verify and benchmark

Run the focused tests, full adapter suite, `compileall`, and `git diff --check`. Replay Stage294 row `88432d0a` with a live judge using frozen Top-20 retrieval; require a valid answer of 4/PASS and no API error. If accepted, run a fresh 500-question answerer/judge candidate over Stage268 frozen Top-200 retrieval with Stage294 settings plus this flag. Compare paired outcomes against Stage291 and Stage294; do not attribute unrelated answer-generation variation to the resolver.

## Task 4: Commit and archive

Commit the code module with an English commit message. Add bilingual scorecard entries with result hashes, trigger count, paired deltas, and unresolved 97%, memory-precision, and disaster-recovery gates. Append the final evidence to the separate Civilization project archive; do not sync secrets or copy this project history into the general memory registry.
