# Collection Cumulative Count Increment Resolution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Include one clearly documented later acquisition when answering a current total for a filtered coin collection.

**Architecture:** Add a default-off resolver restricted to pre-year American coin-collection totals. It requires one user-authored cumulative total and exactly one later user-authored addition to the same explicitly named collection. The added coin must have an explicit year that satisfies the question's cutoff, precede the question date, and be later than the base total in source chronology. A later cumulative total, ambiguous addition set, assistant-only claim, conflicting unit, or unorderable source leaves the answer unchanged. No model call or memory write is added.

**Tech Stack:** Python, existing count evidence models and validators, LongMemEval adapter, unittest.

---

## Evidence and options

Stage283 row `69fee5aa` asks how many pre-1920 American coins the user has. A user quote gives a cumulative total of 37 coins in that collection (May 27). A later user quote (May 29) says they added a 1915-S Barber quarter to the same pre-1920 American coin collection. The answerer returns 37. Another retrieved quote describes a recently bought 1972 doubled-die cent, which does not meet the before-1920 filter and must not be added.

Options considered:

1. **Strict cumulative-plus-one-increment resolver (recommended):** add only one explicitly dated, later, user-authored addition that exactly matches the collection filter. It is narrow and auditable.
2. **General count-inventory expansion:** ask a model to rediscover all collection items across retrieved memories. This broadens inference and can duplicate cumulative values.
3. **Ignore later additions:** preserve the old cumulative total. This leaves a clear post-total acquisition out of the answer.

Keep the resolver opt-in until frozen replay and fresh-run evidence support it.

## Task 1: Add resolver and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Test the 37-total plus 1915 quarter case. Negative cases cover the 1972 coin, assistant-only additions, same-age or unorderable sources, a later cumulative total, multiple qualifying additions, mismatched collection category, and incompatible units. Verify exact user citations, count/unit consistency, and default-off CLI/pipeline behavior.

## Task 2: Implement the opt-in cumulative increment resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--collection-total-include-subsequent-additions`. Parse only a pre-year question and its matching American coin collection, one exact user cumulative total, and one later user-authorized `added`/`acquired` quote with an explicit qualifying coin year. Refuse to act if a newer total exists or the update order is ambiguous. Keep the old total as cumulative evidence, add one incremental item, recompute the total, and validate the answer.

## Task 3: Verify and benchmark

Run focused tests, the full adapter test suite, `compileall`, and `git diff --check`. Replay Stage283's 500 saved answers with the option and judge changed rows; require `69fee5aa` to move FAIL→PASS with zero regressions. Then run a fresh 500-question candidate on the same frozen Stage268 Top-200 retrieval using Stage285 settings plus this option, and record paired outcomes, exact trigger count, retrieval identity, and artifact hashes.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record the frozen replay and full candidate separately in the bilingual scorecard and the separate Civilization project archive, retaining all remaining accuracy and operational gates.
