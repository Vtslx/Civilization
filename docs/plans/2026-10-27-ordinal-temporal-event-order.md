# Ordinal Temporal Event Order Resolution Plan

> **For Codex:** Keep the answer evidence-bound and preserve the 485/500 full Top-20 gate.

**Goal:** Order a complete temporal event list when the question expresses ranks as “first, second, and third” and every event has a source-validated, nonoverlapping date interval.

**Architecture:** Extend the existing default-off `--temporal-event-list-order-resolution` path to recognize explicit ordinal-rank wording in addition to “order/sequence/earliest/latest”. Retain its complete-list, user-authored citation, and pairwise-disjoint interval requirements. Accept only the exact observed typo `yesterady` as a one-character transposition of “yesterday” when deriving an event date from an observation date; do not add fuzzy matching for arbitrary relative-date text.

**Tech Stack:** Python, existing `TemporalEventEvidence` and user-citation validators, pytest, frozen Stage260 temporal answers.

---

## Evidence

Stage260 question `gpt4_7ca326fa` asks who graduated “first, second and third”. Its structured audit lists three user-cited events: Emma, Rachel, and Alex. Rachel and Alex dates resolve to June 21 and a June 27–July 5 interval. Emma's quote says “graduated yesterady” in a conversation dated May 28, 2022; the one-letter transposition prevents the existing observation-relative date parser from resolving May 27. The candidate abstained despite the event order being source-supported; Stage243's saved answer was judged correct.

## Tasks

### Task 1: Add failing tests

**Files:** `memory-benchmarks/tests/test_longmemeval_prompts.py`

- Verify the existing temporal list resolver accepts an explicit three-rank question when the complete user-cited event intervals are strictly ordered.
- Verify the specific `yesterady` transposition resolves to the day before the conversation date.
- Keep ambiguous, incomplete, assistant-only, overlapping, and non-ordinal cases unchanged.

### Task 2: Implement the narrow parser and gate extension

**Files:** `memory-benchmarks/benchmarks/longmemeval/prompts.py`

- Add an ordinal-list predicate requiring a distinct first/second/third rank sequence.
- Accept that predicate alongside the existing list-order question gate.
- Add only `yesterady` as an alias in the observation-relative date parser and map it to the previous calendar day.
- Do not weaken exact quote validation, user-role requirements, expected event counts, or interval separation.

### Task 3: Verify and replay

- Run focused temporal-list tests, the full adapter suite, `compileall`, and `git diff --check`.
- Replay the Stage260 133 temporal saved answers with the existing temporal-list flag; rejudge only changed rows using seed 42.
- Keep and commit only if the frozen replay produces a positive net gain, no regressions, and no invalid citations.
- If it passes, run a fresh 500-question candidate with the matching Stage260 feature set.

## Recovery

The resolver is opt-in and writes no durable memory. Stage260 per-question results remain immutable rollback inputs. Store replay outputs on `/Volumes/Net`; do not put provider credentials or raw prompt payloads in scorecards.
