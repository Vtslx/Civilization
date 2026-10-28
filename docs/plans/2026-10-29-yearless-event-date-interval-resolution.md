# Yearless Event Date Interval Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct event intervals when user-authored quotes give month/day dates without years and the answerer invents a conflicting year.

**Architecture:** Add a default-off deterministic resolver for two-event elapsed-interval questions. It will use exact user citations, a shared observation date that matches the question date, and yearless month/day dates no more than 62 days before that anchor; it will then replace model-inferred years with the source-anchored dates. Ambiguous, future, stale, assistant-authored, or differently anchored dates leave the answer unchanged. No new model call or memory write is added.

**Tech Stack:** Python, existing Pydantic evidence models, LongMemEval adapter, unittest, frozen replay artifacts.

---

## Evidence and options

Stage281 row `08f4fc43` asks how many days passed between two events. User quotes say the events happened on January 2 and February 1, and both source memories are dated February 20, 2023. The answer audit inferred January 2, 2022 and February 1, 2023, so the existing resolver returned 395 days instead of 30. The quotes do not state either year.

Options considered:

1. **Source-anchored month/day resolver (recommended):** deterministically resolve both quoted dates against their common question-date anchor, only for nearby completed events. This has a narrow, auditable evidence boundary.
2. **Prompt-only instruction:** ask the answerer not to invent years. This leaves behavior dependent on model compliance and gives no deterministic correction.
3. **Global temporal-date precedence change:** always prefer source-derived dates over answerer dates. This could alter unrelated temporal ordering and list questions, so it is broader than the observed defect.

The resolver stays opt-in until the frozen temporal replay proves it corrects the target without regressions.

## Task 1: Add focused parser and resolver tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test the January 2 / February 1 example and require a 30-day result with exact user citations. Negative cases cover a different observation anchor, an anchor different from the question date, future or older-than-62-day dates, explicit but conflicting user years, assistant-only citations, ambiguous event matching, and malformed dates. Verify the CLI default is disabled and the flag reaches the answer pipeline and run metadata.

## Task 2: Implement the opt-in source-date resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--yearless-event-date-interval-resolution`. Match each event phrase to one uniquely best cited temporal event. Require one exact user-authored month/day date per event, the same observation date for both citations, and equality between that date and the parsed question date. Resolve each date to its most recent occurrence on or before the anchor and require both occurrences to fall within the preceding 62 days. Compute the requested interval, replace the answer and source citations, and expose an audit status. Keep the flag off by default and preserve all existing resolver behavior when disabled.

## Task 3: Verify and benchmark

Run the focused prompt and pipeline tests, the complete adapter test suite, `compileall`, and `git diff --check`. Replay all 133 Stage281 temporal answers against their frozen Top-20 evidence and judge changed rows. Retain the feature only if `08f4fc43` changes from FAIL to PASS, no previously correct temporal answer regresses, all rewritten citations validate, and every row completes. If accepted, run a fresh all-500 Top-20 candidate over the frozen Stage268 Top-200 retrieval using Stage281 settings plus the new flag; record the paired results and feature trigger count without attributing broad provider variance to this resolver.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record full-run and temporal-slice metrics, artifact hashes, and the remaining 97% and memory-precision gaps in the Civilization scorecard and separate project archive.
