# Relative-time Event Answer Resolution Implementation Plan

> **For Codex:** Implement only with citation-backed date and event identity checks; retain the 485/500 full Top-20 release gate.

**Goal:** Correct temporal entity questions when a unique user-authored event date exactly matches the relative time window but the answerer misses that event.

**Architecture:** Add a default-off deterministic resolver over the existing `TemporalEventEvidence` list. Parse explicit day/week windows from an entity question, derive each event date from an exact user-authored quote and its observation date, and compare that source-derived date with the question date. Rewrite only when exactly one completed event falls in the requested window, the event action or audit relation matches the question, citations validate, and the final answer passes the existing validator. Leave numeric elapsed-duration questions, planned events, assistant-only claims, ambiguous dates, and multiple matches unchanged.

**Tech Stack:** Python, existing LongMemEval temporal event schema, exact-quote citation validators, pytest, frozen Stage243 Top-20 replay.

---

## Evidence

Stage243 contains two temporal entity-retrieval errors with structured, user-cited dates that exactly match the question window:

- `2ebe6c92`: “Which book did I finish a week ago?” The question date is 2023-02-07; a user quote dated 2023-01-31 says they finished *The Nightingale*. Another event in the audit is only a reading goal, so the resolver must reject it as incomplete.
- `eac54add`: “What was the significant business milestone I mentioned four weeks ago?” A user quote dated 2023-03-01 records signing the first-client contract; the question date is 2023-03-28, 27 days later, and the audit explicitly labels that event four weeks before the question date.

The answerer returned an abstention and an unrelated influencer event respectively, despite the source-backed events already appearing in its temporal audit. The resolver must recompute dates from user citations and never trust a model-supplied event date by itself.

## Tasks

### Task 1: Add narrow contract tests

**Files:** `memory-benchmarks/tests/test_longmemeval_prompts.py`, `memory-benchmarks/tests/test_longmemeval_run.py`

- Verify exact day/week matching from a user-authored “today” quote and question date.
- Verify a planned or hoped-for event is excluded even when its date matches.
- Verify assistant-only, invalid, ambiguous, out-of-window, duplicate-event, and numeric-duration cases remain unchanged.
- Verify the new CLI option defaults off and run/per-row metadata records whether the resolver applied.

### Task 2: Implement the opt-in resolver

**Files:** `memory-benchmarks/benchmarks/longmemeval/prompts.py`, `memory-benchmarks/benchmarks/longmemeval/run.py`, `memory-benchmarks/README.md`

- Add `--relative-time-event-answer-resolution`, disabled by default.
- Scope it to nonnumeric temporal entity questions with an explicit day/week “ago” window.
- Use `_user_cited_event_date` to derive exact dates from validated user-authored quotes; reject unresolved, planned, future, or multiply matched events.
- Rewrite from the uniquely matched event's supported description and citations, then run the shared answer validator. Restore the original response on any validation error.

### Task 3: Run a frozen replay and acceptance gate

- Replay all 133 Stage243 temporal answers with the same saved Top-20 evidence and no reranking.
- Rejudge only changed answers with the configured judge and seed 42; compare paired strict scores, validation failures, and regressions.
- Keep and commit only if the replay yields a positive net gain, no regression, and no new invalid evidence or unresolved provider/schema failure.
- If it passes, run a fresh full 500-question candidate before claiming release-score improvement. Preserve the full score gate and report saved-answer replay separately.

## Recovery

The feature is opt-in and has no memory-store write path. Stage243 saved per-question files are immutable rollback inputs. Pilot outputs belong on `/Volumes/Net`, not the nearly full system volume. Never put provider credentials in commands, logs, scorecards, or commits.
