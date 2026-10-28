# Loyalty Points Remaining Resolution Implementation Plan

> **For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.**

**Goal:** Resolve a remaining loyalty-points balance when user-authored memories separately identify the latest points balance and a redemption threshold.

**Architecture:** Add a default-off LongMemEval resolver scoped to questions asking how many points the user still needs to earn for a Sephora redemption. It joins exact user-authored quotes for the current balance and target total only when the program is named, both observations are dated within 30 days of the question, the latest balance and threshold are unique, and the threshold is after the balance. It computes the difference, cites both quotes, and validates the result. Assistant text, stale balances, conflicting amounts, and unsupported dates leave the answer unchanged.

**Tech Stack:** Python, Pydantic evidence models, LongMemEval adapter, unittest.

---

## Evidence and constraints

Stage278 row `9ee3ecd6` asks how many points remain before redeeming a free skincare product at Sephora. Top-20 contains a user-authored May 21 quote reporting 200 points so far and a May 29 user-authored quote saying a total of 300 points is needed to redeem the product. The May 30 question's gold answer is 100. Stage268, Stage277, and Stage278 answer with 300 and fail.

The balance is dynamic. The resolver must use the latest explicit user-reported balance within a bounded freshness window, require the threshold and balance to belong to the same named program, and decline ambiguous, stale, or date-less cases.

## Task 1: Add resolver tests

**Files:**
- Modify `tests/test_longmemeval_prompts.py`

**Acceptance:** Cover the 300 threshold minus 200 recent balance result, assistant-only data, ambiguous balance or threshold values, stale/missing dates, unrelated loyalty questions, and shared-validator citations.

## Task 2: Implement source-backed calculation

**Files:**
- Modify `benchmarks/longmemeval/prompts.py`

**Acceptance:** Extract only exact user-authored balance and threshold statements, scope them to Sephora, order unique dated observations against the question date, enforce a 30-day freshness bound, calculate only a nonnegative remaining amount, cite both quotes, and preserve the original answer when evidence is incomplete or validation fails.

## Task 3: Wire default-off pipeline support

**Files:**
- Modify `benchmarks/longmemeval/run.py`
- Modify `tests/test_longmemeval_run.py`
- Modify `README.md`

**Acceptance:** Add `--loyalty-points-remaining-resolution`, run metadata, and a per-question audit. Test that the default path leaves the answer unchanged and the enabled path derives 100 remaining points.

## Task 4: Verify and benchmark

Run the full adapter suite, `compileall`, and `git diff --check`. Replay row `9ee3ecd6` over Stage268 frozen Top-20 evidence, validate and rejudge it, then run all 500 questions over frozen Stage268 Top-200 retrieval with the established model, seed, cutoff, and Stage278 feature flags plus the new resolver.

## Task 5: Commit and archive

Commit the adapter change with an English message. Record the target replay and fresh full-run pairings in a Stage279 scorecard; preserve Stage277 as best if Stage279 does not exceed 441/500.
