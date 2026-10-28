# Relative Weekday Wake-Time Resolution Plan

> **For Codex:** Implement this plan task by task. Keep the resolver default-off and require user-authored, source-ordered evidence.

**Goal:** Resolve a weekday wake-up time from a later general wake-time update and an earlier explicit weekday offset.

**Architecture:** Add an opt-in deterministic resolver for temporal-reasoning questions. It will combine one user-authored weekday-offset quote with one later user-authored generic wake-time update only when both share the same observation date, source metadata orders them, the weekday sets match the question, and both times are unique. It will cite both exact quotes and abstain from updating when any condition is missing, stale, assistant-authored, or conflicting.

**Tech Stack:** Python 3.10+, existing Pydantic evidence models and citation validators, LongMemEval adapter, unittest.

---

## Evidence and options

Stage286, Stage287, and Stage288 all miss `gpt4_2c50253f`: “What time do I wake up on Tuesdays and Thursdays?” The answerer uses the older 8:30 AM baseline and subtracts 15 minutes, returning 8:15 AM. In the same May 27 observation date, a user quote in source session 21 says they began waking 15 minutes earlier on Tuesdays and Thursdays; a later user quote in source session 43 updates the general wake time to 7:00 AM. The supported result is 6:45 AM.

Options:

1. **Source-ordered, same-day resolver (recommended):** combine the explicit user weekday offset with the later general user wake time; require a unique baseline, exact weekdays, matching observation date, and source order.
2. **Prompt-only arithmetic reminder:** rejected because it has repeatedly selected the stale 8:30 AM baseline across three full runs.
3. **General schedule inference:** rejected because it could extrapolate unstated routines or combine updates from different dates.

## Task 1: Add resolver and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test the exact 7:00 AM minus 15-minute weekday case, default-off behavior, unrelated days, assistant-only evidence, conflicting or non-unique baselines, mismatched observation dates, and source-order requirements. Verify that the rewritten answer cites both user quotes and passes validation.

## Task 2: Implement the opt-in resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--relative-weekday-wake-time-resolution`. Apply only to temporal questions explicitly asking a wake-up time for named weekdays. Accept a single later user-authored general wake-time update and a matching earlier weekday offset from the same observation date, ordered by source session/pair metadata. Parse a bounded 12-hour clock time and minute offset, require an unambiguous result, write a cited answer, revalidate it, and record the effective flag/status. Do not store a token or write memory data.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay the saved Stage288 target with the flag and a live judge; require 8:15 AM/FAIL to change to 6:45 AM/PASS with valid citations and no saved-answer regressions. If accepted, run a fresh 500-question candidate over the frozen Stage268 Top-200 retrieval using Stage288 settings plus the flag, then record paired results and retrieval equality.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record the target replay, fresh score, hashes, and the remaining 97%, memory-precision, and disaster-recovery gaps in the bilingual scorecards and separate Civilization project archive.
