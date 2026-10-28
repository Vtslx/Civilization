# Latest Follower Estimate Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prefer a user's newer approximate follower-count update over an older exact count when answering a current-count question.

**Architecture:** Add a default-off resolver for current follower questions about a named social platform. It may replace an older user-authored exact count only when a later user-authored quote from the same platform gives a unique approximate follower estimate, source ordering proves the update is newer, and no later conflicting update exists. The answer must preserve uncertainty (for example, “about 1,300 followers”) and cite the estimate. Assistant-only suggestions, platform mismatches, ambiguous ordering, and unsupported estimates leave the answer unchanged.

**Tech Stack:** Python, existing Pydantic answer/count evidence models, LongMemEval adapter, unittest.

---

## Evidence and options

Stage283 row `a2f3aa27` asks how many Instagram followers the user has now. The answer cites an exact user statement of 1,250 followers from session index 16. A later user session (index 21, same observation date) says they are “close to 1300” and refers to their current Instagram follower count. The answerer kept 1,250 and failed the benchmark reference of 1,300.

Options considered:

1. **Qualified latest-estimate resolver (recommended):** use the later same-platform user estimate and say it is approximate. This respects both the newer update and its uncertainty.
2. **Keep the older exact count:** avoids interpreting an estimate but ignores the later user statement and fails the benchmark case.
3. **Broad current-count prompt rewrite:** asks the model to reconcile history without a deterministic ordering or audit gate; this is harder to verify and may affect unrelated facts.

The resolver remains opt-in until the frozen target replay passes and a fresh full run records its exact trigger and paired outcomes.

## Task 1: Add focused resolver and pipeline tests

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/tests/test_longmemeval_run.py`

Cover a later same-platform “close to N followers” update overriding an older exact count with an explicitly approximate answer. Negative cases cover assistant-only estimates, a different platform, an older estimate, a later exact update, ambiguous session ordering, and a non-current historical question. Test the default-off flag, pipeline audit status, citation validation, and answer/count audit agreement.

## Task 2: Implement the opt-in latest estimate resolver

**Files:**
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/prompts.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/benchmarks/longmemeval/run.py`
- Modify: `/Users/yikewang/Documents/本机操作/memory-benchmarks/README.md`

Add `--latest-user-follower-count-estimate-resolution`. Parse exact and approximate follower counts only from exact user-authored quotes that name the platform in the user message. Order them by observation date, source-session index, and pair index. Apply only when the current answer matches the older exact user count, the newer estimate is unique, and no later conflicting update exists. Output the estimate with an explicit qualifier and citation; leave every ambiguous or stale case unchanged.

## Task 3: Verify and benchmark

Run focused tests, the full adapter suite, `compileall`, and `git diff --check`. Replay Stage283 row `a2f3aa27` over its frozen Top-20 evidence and judge the qualified estimate. Keep the feature only if the target changes FAIL→PASS with valid citations and no frozen-slice regression. Then run a fresh 500-question answerer-and-judge candidate over Stage268 frozen Top-200 retrieval using Stage283 settings plus this flag, and record exact triggers, pairings, and retrieval equality.

## Task 4: Commit and archive

Commit the adapter module with an English message. Record the frozen replay and full candidate separately in the bilingual scorecard and Civilization project archive; retain the 97%, 100% precision, and disaster-recovery gates as open until independently met.
